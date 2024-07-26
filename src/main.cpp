#include "llama.h"
#include <iostream>
#include <string>

int main() {
  // Initialize llama.cpp context parameters
  llama_context_params params = llama_context_default_params();
  params.n_ctx = 2048;
  params.n_parts = -1;
  params.seed = 1234;
  params.f16_kv = true;
  params.use_mlock = false;

  // Replace this with the path to your model file
  const std::string model_path = "/path/to/your/llama/model.bin";

  // Load the model
  llama_model* model = llama_load_model_from_file(model_path.c_str(), params);

  if (!model) {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  // Create context
  llama_context* ctx = llama_new_context_with_model(model, params);

  if (!ctx) {
    std::cerr << "Failed to create context\n";
    llama_free_model(model);
    return 1;
  }

  // Simple prompt
  const char* prompt = "Hello, world! This is llama.cpp:";
  llama_eval(ctx, llama_tokenize(ctx, prompt, -1, true), 0, 0);

  // Generate some tokens
  for (int i = 0; i < 50; ++i) {
    float* logits = llama_get_logits(ctx);
    int token = llama_sample_top_p_top_k(ctx, nullptr, 0, 40, 0.9f, 0.0f, 0.0f);
    llama_eval(ctx, &token, 1, llama_get_kv_cache_token_count(ctx), 0);

    const char* token_str = llama_token_to_str(ctx, token);
    std::cout << token_str;
  }

  std::cout << std::endl;

  // Clean up
  llama_free(ctx);
  llama_free_model(model);

  return 0;
}
