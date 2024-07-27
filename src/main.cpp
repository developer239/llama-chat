#include "common.h"
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

int main() {
  gpt_params params;
  params.model = "../models/Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf";
  params.prompt = "How do I write hello world in javascript?";
  params.n_predict = 1000;

  llama_backend_init();

  llama_model_params model_params = llama_model_params_from_gpt_params(params);
  llama_model* model = llama_load_model_from_file(params.model.c_str(), model_params);
  if (model == NULL) {
    fprintf(stderr, "Error: unable to load model\n");
    return 1;
  }

  llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
  ctx_params.n_ctx = 80000;
  llama_context* ctx = llama_new_context_with_model(model, ctx_params);
  if (ctx == NULL) {
    fprintf(stderr, "Error: failed to create the llama_context\n");
    return 1;
  }

  std::vector<llama_token> tokens_list = ::llama_tokenize(ctx, params.prompt, true);

  llama_batch batch = llama_batch_init(512, 0, 1);
  for (size_t i = 0; i < tokens_list.size(); i++) {
    llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
  }
  batch.logits[batch.n_tokens - 1] = true;

  if (llama_decode(ctx, batch) != 0) {
    fprintf(stderr, "Error: llama_decode() failed\n");
    return 1;
  }

  int n_cur = batch.n_tokens;
  printf("%s", params.prompt.c_str());  // Print the prompt
  fflush(stdout);

  while (n_cur <= params.n_predict) {
    auto n_vocab = llama_n_vocab(model);
    auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
      candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

    if (llama_token_is_eog(model, new_token_id) || n_cur == params.n_predict) {
      break;
    }

    printf("%s", llama_token_to_piece(ctx, new_token_id).c_str());
    fflush(stdout);

    llama_batch_clear(batch);
    llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
    n_cur += 1;

    if (llama_decode(ctx, batch)) {
      fprintf(stderr, "Error: failed to eval\n");
      return 1;
    }
  }

  printf("\n");

  llama_batch_free(batch);
  llama_free(ctx);
  llama_free_model(model);
  llama_backend_free();

  return 0;
}
