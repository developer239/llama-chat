#include "llama-wrapper.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#include "common.h"
#include "llama.h"

class LlamaWrapper::Impl {
 public:
  Impl() { llama_backend_init(); }

  ~Impl() {
    // Smart pointers with custom deleters handle cleanup
    llama_backend_free();
  }

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

    // Get the token ID for <|eot_id|>
    auto eot_tokens = Encode("<|eot_id|>", false, true);  // Set parseSpecial to true
    if (eot_tokens.size() != 1) {
      std::cerr << "Failed to retrieve <|eot_id|> token ID." << std::endl;
      return false;
    }
    eotToken = eot_tokens[0].tokenId;

    return true;
  }

  [[nodiscard]] std::vector<LlamaToken> Encode(
      const std::string& text, bool addBos, bool parseSpecial = false
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
        parseSpecial  // Pass the parseSpecial parameter here
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
      const std::string& prompt, const SamplingParams& params,
      const std::function<void(const std::string&)>& callback,
      bool addBos
  ) const {
    // Tokenize the prompt with parseSpecial set to true
    auto tokens = Encode(prompt, addBos, true);

    // Initialize the batch
    llama_batch batch = llama_batch_init(params.maxTokens, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      llama_batch_add(batch, tokens[i].tokenId, i, {0}, false);
    }

    if (llama_decode(ctx.get(), batch) != 0) {
      throw std::runtime_error("llama_decode() failed");
    }

    size_t nCur = batch.n_tokens;
    while (nCur < params.maxTokens) {
      auto new_token = SampleToken(params);

      // Check if the generated token is <|eot_id|>
      if (new_token.tokenId == eotToken) break;

      std::string piece = llama_token_to_piece(ctx.get(), new_token.tokenId);
      callback(piece);

      llama_batch_clear(batch);
      llama_batch_add(batch, new_token.tokenId, nCur, {0}, true);
      nCur += 1;

      if (llama_decode(ctx.get(), batch) != 0) {
        throw std::runtime_error("Failed to evaluate");
      }
    }

    llama_batch_free(batch);
  }

 private:
  // Custom deleters for smart pointers
  struct LlamaModelDeleter {
    void operator()(llama_model* model) const { llama_free_model(model); }
  };

  struct LlamaContextDeleter {
    void operator()(llama_context* ctx) const { llama_free(ctx); }
  };

  std::unique_ptr<llama_model, LlamaModelDeleter> model = nullptr;
  std::unique_ptr<llama_context, LlamaContextDeleter> ctx = nullptr;

  llama_token eotToken;

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
    const std::string& prompt, const SamplingParams& params,
    const std::function<void(const std::string&)>& callback, bool addBos
) const {
  pimpl->RunQueryStream(prompt, params, callback, addBos);
}
