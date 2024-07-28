#include "llama-wrapper.h"

#include "common.h"
#include "llama.h"

class LlamaWrapper::Impl {
 public:
  Impl() { llama_backend_init(); }

  ~Impl() {
    if (ctx) llama_free(ctx);
    if (model) llama_free_model(model);
    llama_backend_free();
  }

  bool InitializeModel(
      const std::string& model_path, const ModelParams& params
  ) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_params.vocab_only = params.vocab_only;
    model_params.use_mmap = params.use_mmap;
    model_params.use_mlock = params.use_mlock;

    model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model) {
      throw std::runtime_error("Failed to load model from " + model_path);
    }
    return true;
  }

  bool InitializeContext(const ContextParams& params) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_batch = params.n_batch;
    ctx_params.logits_all = params.logits_all;
    ctx_params.embeddings = params.embedding;

    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
      throw std::runtime_error("Failed to create the llama_context");
    }
    return true;
  }

  std::vector<LlamaToken> Encode(const std::string& text, bool add_bos) const {
    auto llama_tokens = llama_tokenize(ctx, text, add_bos);
    std::vector<LlamaToken> tokens;
    for (auto token : llama_tokens) {
      tokens.emplace_back(token);
    }
    return tokens;
  }

  std::string Decode(const std::vector<LlamaToken>& tokens) const {
    std::string result;
    for (const auto& token : tokens) {
      result += llama_token_to_piece(ctx, token.token_id);
    }
    return result;
  }

  LlamaToken TokenBos() const { return LlamaToken(llama_token_bos(model)); }
  LlamaToken TokenEos() const { return LlamaToken(llama_token_eos(model)); }
  LlamaToken TokenNl() const { return LlamaToken(llama_token_nl(model)); }

  std::string RunQuery(
      const std::string& prompt, const SamplingParams& params,
      bool add_bos = true
  ) const {
    auto tokens = Encode(prompt, add_bos);
    std::string result;

    llama_batch batch = llama_batch_init(params.max_tokens, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      llama_batch_add(batch, tokens[i].token_id, i, {0}, false);
    }

    if (llama_decode(ctx, batch) != 0) {
      throw std::runtime_error("llama_decode() failed");
    }

    size_t n_cur = batch.n_tokens;
    while (n_cur < params.max_tokens) {
      auto new_token = SampleToken(params);
      if (llama_token_is_eog(model, new_token.token_id)) break;

      result += llama_token_to_piece(ctx, new_token.token_id);
      llama_batch_clear(batch);
      llama_batch_add(batch, new_token.token_id, n_cur, {0}, true);
      n_cur += 1;

      if (llama_decode(ctx, batch) != 0) {
        throw std::runtime_error("Failed to evaluate");
      }
    }

    llama_batch_free(batch);
    return result;
  }

  void RunQueryStream(
      const std::string& prompt, const SamplingParams& params,
      const std::function<void(const std::string&)>& callback,
      bool add_bos = true
  ) const {
    auto tokens = Encode(prompt, add_bos);

    llama_batch batch = llama_batch_init(params.max_tokens, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
      llama_batch_add(batch, tokens[i].token_id, i, {0}, false);
    }

    if (llama_decode(ctx, batch) != 0) {
      throw std::runtime_error("llama_decode() failed");
    }

    size_t n_cur = batch.n_tokens;
    while (n_cur < params.max_tokens) {
      auto new_token = SampleToken(params);
      if (llama_token_is_eog(model, new_token.token_id)) break;

      std::string piece = llama_token_to_piece(ctx, new_token.token_id);
      callback(piece);

      llama_batch_clear(batch);
      llama_batch_add(batch, new_token.token_id, n_cur, {0}, true);
      n_cur += 1;

      if (llama_decode(ctx, batch) != 0) {
        throw std::runtime_error("Failed to evaluate");
      }
    }

    llama_batch_free(batch);
  }

 private:
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;

  LlamaToken SampleToken(const SamplingParams& params) const {
    auto logits = llama_get_logits(ctx);
    auto n_vocab = llama_n_vocab(model);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
      candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f}
      );
    }

    llama_token_data_array candidates_p = {
        candidates.data(),
        candidates.size(),
        false
    };

    if (!params.repeat_penalty_tokens.empty()) {
      std::vector<llama_token> penalty_tokens;
      for (const auto& token : params.repeat_penalty_tokens) {
        penalty_tokens.push_back(token.token_id);
      }

      llama_sample_repetition_penalties(
          ctx,
          &candidates_p,
          penalty_tokens.data(),
          penalty_tokens.size(),
          params.repeat_penalty,
          params.frequency_penalty,
          params.presence_penalty
      );
    }

    llama_sample_top_k(ctx, &candidates_p, params.top_k, 1);
    llama_sample_top_p(ctx, &candidates_p, params.top_p, 1);
    llama_sample_temp(ctx, &candidates_p, params.temperature);

    return LlamaToken(llama_sample_token(ctx, &candidates_p));
  }
};

// Implement LlamaWrapper methods
LlamaWrapper::LlamaWrapper() : pimpl(std::make_unique<Impl>()) {}
LlamaWrapper::~LlamaWrapper() = default;

bool LlamaWrapper::InitializeModel(
    const std::string& model_path, const ModelParams& params
) {
  return pimpl->InitializeModel(model_path, params);
}

bool LlamaWrapper::InitializeContext(const ContextParams& params) {
  return pimpl->InitializeContext(params);
}

std::vector<LlamaToken> LlamaWrapper::Encode(
    const std::string& text, bool add_bos
) const {
  return pimpl->Encode(text, add_bos);
}

std::string LlamaWrapper::Decode(const std::vector<LlamaToken>& tokens) const {
  return pimpl->Decode(tokens);
}

LlamaToken LlamaWrapper::TokenBos() const { return pimpl->TokenBos(); }
LlamaToken LlamaWrapper::TokenEos() const { return pimpl->TokenEos(); }
LlamaToken LlamaWrapper::TokenNl() const { return pimpl->TokenNl(); }

std::string LlamaWrapper::RunQuery(
    const std::string& prompt, const SamplingParams& params, bool add_bos
) const {
  return pimpl->RunQuery(prompt, params, add_bos);
}

void LlamaWrapper::RunQueryStream(
    const std::string& prompt, const SamplingParams& params,
    const std::function<void(const std::string&)>& callback, bool add_bos
) const {
  pimpl->RunQueryStream(prompt, params, callback, add_bos);
}
