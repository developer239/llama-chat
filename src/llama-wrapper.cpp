#include "llama-wrapper.h"

#include <iostream>
#include <map>
#include <stdexcept>

#include "common.h"
#include "llama.h"

class LlamaWrapper::Impl {
 public:
  Impl() { llama_backend_init(); }

  ~Impl() {
    if (ctx) {
      llama_free(ctx);
    }
    if (model) {
      llama_free_model(model);
    }

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

    model = llama_load_model_from_file(model_path.c_str(), modelParams);
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

    ctx = llama_new_context_with_model(model, ctxParams);
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
        model,
        text.c_str(),
        text.length(),
        llamaTokens.data(),
        maxTokens,
        addBos,
        /*parse_special=*/false
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

  [[nodiscard]] std::string Decode(const std::vector<LlamaToken>& tokens
  ) const {
    std::string result;
    for (const auto& token : tokens) {
      result += llama_token_to_piece(
          ctx,
          token.tokenId,
          // TODO: should add or not?
          false
      );
    }
    return result;
  }

  [[nodiscard]] LlamaToken TokenBos() const {
    return LlamaToken(llama_token_bos(model));
  }
  [[nodiscard]] LlamaToken TokenEos() const {
    return LlamaToken(llama_token_eos(model));
  }
  [[nodiscard]] LlamaToken TokenNl() const {
    return LlamaToken(llama_token_nl(model));
  }

  void RunQueryStream(
      const std::string& systemPrompt, const std::string& userMessage,
      const SamplingParams& params,
      const std::function<void(const std::string&)>& callback
  ) const {
    // Simplified prompt without unrecognized special tokens
    std::string fullPrompt = "system\n" + systemPrompt + "\n" + "user\n" +
                             userMessage + "\n" + "assistant";

    // Set 'addBos' to true to add the BOS token during tokenization
    auto tokens = Encode(fullPrompt, true);

    if (tokens.empty()) {
      std::cerr << "Tokenization of the prompt failed." << std::endl;
      return;
    }

    llama_batch batch = llama_batch_init(params.maxTokens, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      llama_batch_add(batch, tokens[i].tokenId, i, {0}, false);
    }

    if (llama_decode(ctx, batch) != 0) {
      std::cerr << "llama_decode() failed" << std::endl;
      return;
    }

    size_t nCur = batch.n_tokens;
    while (nCur < params.maxTokens) {
      auto newToken = SampleToken(params);
      if (newToken.tokenId == llama_token_eos(model)) {
        break;
      }

      std::string piece = llama_token_to_piece(
          ctx,
          newToken.tokenId,
          // TODO: should add or not?
          false
      );
      callback(piece);

      llama_batch_clear(batch);
      llama_batch_add(batch, newToken.tokenId, nCur, {0}, true);
      nCur += 1;

      if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Failed to evaluate" << std::endl;
        return;
      }
    }

    llama_batch_free(batch);
  }

 private:
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;

  std::map<std::string, llama_token> specialTokenIds;

  void InitializeSpecialTokens() {
    specialTokenIds["bos"] = llama_token_bos(model);
    specialTokenIds["eos"] = llama_token_eos(model);
    specialTokenIds["nl"] = llama_token_nl(model);
    // Add other tokens if they are available
  }

  [[nodiscard]] LlamaToken SampleToken(const SamplingParams& params) const {
    const float* logits = llama_get_logits(ctx);
    const int nVocabulary = llama_n_vocab(model);

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
          ctx,
          &candidatesP,
          penaltyTokens.data(),
          penaltyTokens.size(),
          params.repeatPenalty,
          params.frequencyPenalty,
          params.presencePenalty
      );
    }

    llama_sample_top_k(ctx, &candidatesP, params.topK, 1);
    llama_sample_top_p(ctx, &candidatesP, params.topP, 1);
    llama_sample_temp(ctx, &candidatesP, params.temperature);

    return LlamaToken(llama_sample_token(ctx, &candidatesP));
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

std::string LlamaWrapper::Decode(const std::vector<LlamaToken>& tokens) const {
  return pimpl->Decode(tokens);
}

LlamaToken LlamaWrapper::TokenBos() const { return pimpl->TokenBos(); }
LlamaToken LlamaWrapper::TokenEos() const { return pimpl->TokenEos(); }
LlamaToken LlamaWrapper::TokenNl() const { return pimpl->TokenNl(); }

void LlamaWrapper::RunQueryStream(
    const std::string& systemPrompt, const std::string& userMessage,
    const SamplingParams& params,
    const std::function<void(const std::string&)>& callback
) const {
  pimpl->RunQueryStream(systemPrompt, userMessage, params, callback);
}
