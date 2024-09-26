#include "llama-chat.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common.h"
#include "llama.h"

class LlamaChat::Impl {
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

    auto eot_tokens = Encode("<|eot_id|>", false, true);
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
    int maxTokens = text.length() + (addBos ? 1 : 0);
    std::vector<llama_token> llamaTokens(maxTokens);

    int nTokens = llama_tokenize(
        model.get(),
        text.c_str(),
        text.length(),
        llamaTokens.data(),
        maxTokens,
        addBos,
        parseSpecial
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

  void Prompt(
      const std::string& userMessage, const SamplingParams& params,
      const std::function<void(const std::string&)>& callback
  ) {
    AddUserMessage(userMessage);
    RunQueryStream(params, [this, &callback](const std::string& piece) {
      if (!IsSpecialToken(piece)) {
        callback(piece);
      }
    });
  }

  void SetSystemPrompt(const std::string& systemPrompt) {
    conversationHistory.clear();
    conversationHistory.push_back({"system", systemPrompt});
  }

  void ResetConversation() { conversationHistory.clear(); }

 private:
  struct LlamaModelDeleter {
    void operator()(llama_model* model) const { llama_free_model(model); }
  };

  struct LlamaContextDeleter {
    void operator()(llama_context* ctx) const { llama_free(ctx); }
  };

  struct Message {
    std::string role;
    std::string content;
  };

  std::vector<Message> conversationHistory;
  std::unique_ptr<llama_model, LlamaModelDeleter> model = nullptr;
  std::unique_ptr<llama_context, LlamaContextDeleter> ctx = nullptr;
  llama_token eotToken;

  void BuildPrompt(std::string& prompt) const {
    std::ostringstream oss;
    oss << "<|begin_of_text|>";
    for (const auto& msg : conversationHistory) {
      oss << "<|start_header_id|>" << msg.role << "<|end_header_id|>"
          << msg.content << "<|eot_id|>";
    }
    oss << "<|start_header_id|>assistant<|end_header_id|>";
    prompt = oss.str();
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

  void AddUserMessage(const std::string& message) {
    conversationHistory.push_back({"user", message});
  }

  void RunQueryStream(
      const SamplingParams& params,
      const std::function<void(const std::string&)>& callback
  ) {
    std::string prompt;
    BuildPrompt(prompt);

    auto tokens = Encode(prompt, false, true);

    llama_batch batch = llama_batch_init(params.maxTokens, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      llama_batch_add(batch, tokens[i].tokenId, i, {0}, false);
    }

    if (llama_decode(ctx.get(), batch) != 0) {
      throw std::runtime_error("llama_decode() failed");
    }

    size_t nCur = batch.n_tokens;
    std::string assistantResponse;

    std::string currentPiece;
    while (nCur < params.maxTokens) {
      auto new_token = SampleToken(params);

      if (new_token.tokenId == eotToken) break;

      std::string piece = llama_token_to_piece(ctx.get(), new_token.tokenId);
      currentPiece += piece;

      if (!IsSpecialToken(currentPiece) && !currentPiece.empty()) {
        callback(currentPiece);
        assistantResponse += currentPiece;
        currentPiece.clear();
      }

      llama_batch_clear(batch);
      llama_batch_add(batch, new_token.tokenId, nCur, {0}, true);
      nCur += 1;

      if (llama_decode(ctx.get(), batch) != 0) {
        throw std::runtime_error("Failed to evaluate");
      }
    }

    llama_batch_free(batch);
    conversationHistory.push_back({"assistant", assistantResponse});
  }

  bool IsSpecialToken(const std::string& piece) const {
    std::vector<std::string> specialTokens = {
        "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>",
        "<|end_header_id|>", "<|eot_id|>"
    };
    for (const auto& token : specialTokens) {
      if (piece.find(token) != std::string::npos) {
        return true;
      }
    }
    return false;
  }
};

LlamaChat::LlamaChat() : pimpl(std::make_unique<Impl>()) {}
LlamaChat::~LlamaChat() = default;

bool LlamaChat::InitializeModel(
    const std::string& modelPath, const ModelParams& params
) {
  try {
    return pimpl->InitializeModel(modelPath, params);
  } catch (const std::exception& e) {
    std::cerr << "InitializeModel exception: " << e.what() << std::endl;
    return false;
  }
}

bool LlamaChat::InitializeContext(const ContextParams& params) {
  try {
    return pimpl->InitializeContext(params);
  } catch (const std::exception& e) {
    std::cerr << "InitializeContext exception: " << e.what() << std::endl;
    return false;
  }
}

void LlamaChat::SetSystemPrompt(const std::string& systemPrompt) {
  pimpl->SetSystemPrompt(systemPrompt);
}

void LlamaChat::ResetConversation() { pimpl->ResetConversation(); }

void LlamaChat::Prompt(
    const std::string& userMessage, const SamplingParams& params,
    const std::function<void(const std::string&)>& callback
) {
  return pimpl->Prompt(userMessage, params, callback);
}

std::vector<LlamaToken> LlamaChat::Encode(const std::string& text, bool addBos)
    const {
  return pimpl->Encode(text, addBos);
}
