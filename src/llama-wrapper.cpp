#include "llama-wrapper.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <stdexcept>
#include <iostream>

class LlamaWrapper::Impl {
 public:
  Impl() {
    llama_backend_init();
  }

  ~Impl() {
    if (ctx) llama_free(ctx);
    if (model) llama_free_model(model);
    llama_backend_free();
  }

  bool Initialize(const std::string& model_path, size_t context_size) {
    gpt_params params;
    params.model = model_path;
    params.n_ctx = context_size;

    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (!model) {
      throw std::runtime_error("Error: unable to load model from " + model_path);
    }

    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
      llama_free_model(model);
      throw std::runtime_error("Error: failed to create the llama_context");
    }
    return true;
  }

  std::string RunQuery(const std::string& prompt, size_t max_tokens) const {
    if (!ctx || !model) throw std::runtime_error("Model not initialized");

    std::vector<llama_token> tokens_list = llama_tokenize(ctx, prompt, true);
    if (tokens_list.size() + max_tokens > llama_n_ctx(ctx)) {
      throw std::runtime_error("Request exceeds model's context size");
    }

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < tokens_list.size(); ++i) {
      llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    batch.logits[batch.n_tokens - 1] = true;  // Ensure logits for last token
    if (llama_decode(ctx, batch) != 0) {
      throw std::runtime_error("llama_decode() failed");
    }

    std::string result;
    size_t n_cur = batch.n_tokens;

    while (n_cur < max_tokens) {
      auto n_vocab = llama_n_vocab(model);
      auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
      if (!logits) {
        throw std::runtime_error("Failed to get logits");
      }

      std::vector<llama_token_data> candidates;
      candidates.reserve(n_vocab);
      for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
      }

      llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
      const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

      if (llama_token_is_eog(model, new_token_id) || n_cur == max_tokens) {
        break;
      }

      result += llama_token_to_piece(ctx, new_token_id);
      llama_batch_clear(batch);
      llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
      n_cur += 1;

      if (llama_decode(ctx, batch)) {
        throw std::runtime_error("Failed to evaluate");
      }
    }

    llama_batch_free(batch);
    return result;
  }

  void RunQueryStream(const std::string& prompt, size_t max_tokens, const std::function<void(const std::string&)>& callback) const {
    if (!ctx || !model) throw std::runtime_error("Model not initialized");

    std::vector<llama_token> tokens_list = llama_tokenize(ctx, prompt, true);
    if (tokens_list.size() + max_tokens > llama_n_ctx(ctx)) {
      throw std::runtime_error("Request exceeds model's context size");
    }

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < tokens_list.size(); ++i) {
      llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    batch.logits[batch.n_tokens - 1] = true;  // Ensure logits for last token
    if (llama_decode(ctx, batch) != 0) {
      throw std::runtime_error("llama_decode() failed");
    }

    size_t n_cur = batch.n_tokens;

    while (n_cur < max_tokens) {
      auto n_vocab = llama_n_vocab(model);
      auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
      if (!logits) {
        throw std::runtime_error("Failed to get logits");
      }

      std::vector<llama_token_data> candidates;
      candidates.reserve(n_vocab);
      for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
      }

      llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
      const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

      if (llama_token_is_eog(model, new_token_id) || n_cur == max_tokens) {
        break;
      }

      std::string piece = llama_token_to_piece(ctx, new_token_id);
      callback(piece);

      llama_batch_clear(batch);
      llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
      n_cur += 1;

      if (llama_decode(ctx, batch)) {
        throw std::runtime_error("Failed to evaluate");
      }
    }

    llama_batch_free(batch);
  }

 private:
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
};

LlamaWrapper::LlamaWrapper() : pimpl(std::make_unique<Impl>()) {}
LlamaWrapper::~LlamaWrapper() = default;

bool LlamaWrapper::Initialize(const std::string& model_path, size_t context_size) {
  return pimpl->Initialize(model_path, context_size);
}

std::string LlamaWrapper::RunQuery(const std::string& prompt, size_t max_tokens) const {
  return pimpl->RunQuery(prompt, max_tokens);
}

void LlamaWrapper::RunQueryStream(const std::string& prompt, size_t max_tokens, const std::function<void(const std::string&)>& callback) const {
  pimpl->RunQueryStream(prompt, max_tokens, callback);
}
