#include "llama-wrapper.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

#include "common.h"

class LlamaWrapper::Impl {
 public:
  Impl() { llama_backend_init(); }

  ~Impl() { llama_backend_free(); }

  bool InitializeModel(
      const std::string& model_path, const ModelParams& params
  ) {
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = params.nGpuLayers;
    modelParams.vocab_only = params.vocabularyOnly;
    modelParams.use_mmap = params.useMemoryMapping;
    modelParams.use_mlock = params.useModelLock;

    model.reset(llama_load_model_from_file(model_path.c_str(), modelParams));
    if (!model) {
      std::cerr << "Failed to load model from " << model_path << std::endl;
      return false;
    }

    return true;
  }

  bool InitializeContext(const ContextParams& params) {
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = params.nContext;
    ctxParams.n_threads = params.nThreads;
    ctxParams.n_batch = params.nBatch;
    ctxParams.logits_all = false;
    ctxParams.embeddings = false;

    ctx.reset(llama_new_context_with_model(model.get(), ctxParams));
    if (!ctx) {
      std::cerr << "Failed to create the llama_context" << std::endl;
      return false;
    }

    InitializeSpecialTokens();

    return true;
  }

  [[nodiscard]] std::vector<LlamaToken> Encode(
      const std::string& text, bool addBos
  ) const {
    // Estimate the maximum number of tokens
    int maxTokens = text.length() + (addBos ? 1 : 0);

    std::vector<llama_token> llamaTokens(maxTokens);

    int nTokens = llama_tokenize(
        model.get(),
        text.c_str(),
        text.length(),
        llamaTokens.data(),
        maxTokens,
        addBos,
        false
    );

    if (nTokens < 0) {
      std::cerr << "Tokenization failed with error code: " << nTokens
                << std::endl;
      return {};
    }

    llamaTokens.resize(nTokens);

    std::vector<LlamaToken> tokens;
    tokens.reserve(nTokens);

    for (auto token : llamaTokens) {
      tokens.emplace_back(token);
    }

    return tokens;
  }

  void RunQueryStream(
      const std::string& systemPrompt,
      const std::vector<std::pair<std::string, std::string>>& conversationHistory,
      const SamplingParams& params,
      const std::function<void(const std::string&)>& callback
  ) const {
    // Build the full prompt from system prompt and conversation history
    std::string fullPrompt = systemPrompt + "\n\n";
    for (const auto& [role, message] : conversationHistory) {
      if (role == "user") {
        fullPrompt += "User: " + message + "\n";
      } else if (role == "assistant") {
        fullPrompt += "Assistant: " + message + "\n";
      }
    }
    fullPrompt += "Assistant: ";

    // Tokenize the prompt
    auto tokens = Encode(fullPrompt, true);

    if (tokens.empty()) {
      std::cerr << "Tokenization of the prompt failed." << std::endl;
      return;
    }

    // Initialize the batch
    llama_batch batch = llama_batch_init(params.maxTokens, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      llama_batch_add(batch, tokens[i].tokenId, i, {0}, false);
    }

    if (llama_decode(ctx.get(), batch) != 0) {
      std::cerr << "llama_decode() failed" << std::endl;
      return;
    }

    size_t nCur = batch.n_tokens;
    std::string generatedText;
    bool stopGeneration = false;

    while (nCur < params.maxTokens && !stopGeneration) {
      auto newToken = SampleToken(params);
      if (newToken.tokenId == llama_token_eos(model.get())) {
        break;
      }

      std::string piece = llama_token_to_piece(
          ctx.get(),
          newToken.tokenId,
          /*special=*/false  // Exclude special tokens in output
      );

      generatedText += piece;

      // Check for stop conditions
      if (generatedText.find("\nUser:") != std::string::npos ||
          generatedText.find("\nAssistant:") != std::string::npos) {
        // Remove the stop sequence from the output
        size_t pos = generatedText.find("\nUser:");
        if (pos == std::string::npos) {
          pos = generatedText.find("\nAssistant:");
        }
        generatedText = generatedText.substr(0, pos);
        stopGeneration = true;
      } else {
        // Output the piece to the callback
        callback(piece);
      }

      llama_batch_clear(batch);
      llama_batch_add(batch, newToken.tokenId, nCur, {0}, true);
      nCur += 1;

      if (llama_decode(ctx.get(), batch) != 0) {
        std::cerr << "Failed to evaluate" << std::endl;
        return;
      }
    }

    llama_batch_free(batch);
  }

 private:
  struct LlamaModelDeleter {
    void operator()(llama_model* model) const { llama_free_model(model); }
  };

  struct LlamaContextDeleter {
    void operator()(llama_context* ctx) const { llama_free(ctx); }
  };

  std::unique_ptr<llama_model, LlamaModelDeleter> model = nullptr;
  std::unique_ptr<llama_context, LlamaContextDeleter> ctx = nullptr;

  std::map<std::string, llama_token> specialTokenIds;

  void InitializeSpecialTokens() {
    specialTokenIds["bos"] = llama_token_bos(model.get());
    specialTokenIds["eos"] = llama_token_eos(model.get());
    specialTokenIds["nl"] = llama_token_nl(model.get());
  }

  [[nodiscard]] LlamaToken SampleToken(const SamplingParams& params) const {
    const float* logits = llama_get_logits(ctx.get());
    const int nVocabulary = llama_n_vocab(model.get());

    std::vector<llama_token_data> candidates;
    candidates.reserve(nVocabulary);

    for (llama_token tokenId = 0; tokenId < nVocabulary; tokenId++) {
      candidates.emplace_back(llama_token_data{tokenId, logits[tokenId], 0.0f});
    }

    llama_token_data_array candidatesP = {
        candidates.data(),
        candidates.size(),
        false
    };

    if (!params.repeatPenaltyTokens.empty()) {
      std::vector<llama_token> penaltyTokens;
      penaltyTokens.reserve(params.repeatPenaltyTokens.size());
      for (const auto& token : params.repeatPenaltyTokens) {
        penaltyTokens.push_back(token.tokenId);
      }

      llama_sample_repetition_penalties(
          ctx.get(),
          &candidatesP,
          penaltyTokens.data(),
          penaltyTokens.size(),
          params.repeatPenalty,
          params.frequencyPenalty,
          params.presencePenalty
      );
    }

    llama_sample_top_k(ctx.get(), &candidatesP, params.topK, 1);
    llama_sample_top_p(ctx.get(), &candidatesP, params.topP, 1);
    llama_sample_temp(ctx.get(), &candidatesP, params.temperature);

    return LlamaToken(llama_sample_token(ctx.get(), &candidatesP));
  }
};

LlamaWrapper::LlamaWrapper() : pimpl(std::make_unique<Impl>()) {}
LlamaWrapper::~LlamaWrapper() = default;

bool LlamaWrapper::InitializeModel(
    const std::string& modelPath, const ModelParams& params
) {
  try {
    return pimpl->InitializeModel(modelPath, params);
  } catch (const std::exception& e) {
    std::cerr << "InitializeModel exception: " << e.what() << std::endl;
    return false;
  }
}

bool LlamaWrapper::InitializeContext(const ContextParams& params) {
  try {
    return pimpl->InitializeContext(params);
  } catch (const std::exception& e) {
    std::cerr << "InitializeContext exception: " << e.what() << std::endl;
    return false;
  }
}

std::vector<LlamaToken> LlamaWrapper::Encode(
    const std::string& text, bool addBos
) const {
  return pimpl->Encode(text, addBos);
}

void LlamaWrapper::RunQueryStream(
    const std::string& systemPrompt,
    const std::vector<std::pair<std::string, std::string>>& conversationHistory,
    const SamplingParams& params,
    const std::function<void(const std::string&)>& callback
) const {
  pimpl->RunQueryStream(systemPrompt, conversationHistory, params, callback);
}
