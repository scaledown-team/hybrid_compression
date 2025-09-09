import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig

class ScaleDownCompressorConfig(PretrainedConfig):
    """
    Configuration class for ScaleDownCompressor.
    """
    model_type = "scaledown_compressor"

    def __init__(
        self,
        compressor_name_or_path="meta-llama/Llama-2-7b-hf",
        # ScaleDown-N-Layers
        num_compressor_layers=8,
        use_n_layers=False,
        # ScaleDown-llama
        generator_hidden_size=4096,
        # Reranking
        add_reranking_head=True,
        **kwargs
    ):
        self.compressor_name_or_path = compressor_name_or_path
        self.num_compressor_layers = num_compressor_layers
        self.use_n_layers = use_n_layers
        self.generator_hidden_size = generator_hidden_size
        self.add_reranking_head = add_reranking_head
        super().__init__(**kwargs)

class ScaleDownCompressor(PreTrainedModel):
    """
    ScaleDown Compressor model.
    This model takes a query and a document and compresses the document
    into a sequence of embedding vectors.
    """
    config_class = ScaleDownCompressorConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.compressor = AutoModel.from_pretrained(config.compressor_name_or_path)

        if config.use_n_layers:
            # ScaleDown-N-Layers: Use the first N layers of the generator as the compressor
            self.compressor.layers = self.compressor.layers[:config.num_compressor_layers]
        else:
            # ScaleDown-llama: Use a smaller LLM as the compressor
            compressor_hidden_size = self.compressor.config.hidden_size
            self.mapping = nn.Sequential(
                nn.Linear(compressor_hidden_size, config.generator_hidden_size),
                nn.ReLU(),
                nn.Linear(config.generator_hidden_size, config.generator_hidden_size),
            )

        if config.add_reranking_head:
            compressor_hidden_size = self.compressor.config.hidden_size
            self.rerank_head = nn.Linear(compressor_hidden_size, 1)


    def forward(self, input_ids, attention_mask, memory_token_indices, rerank_token_index=None):
        """
        Forward pass for the compressor.

        Args:
            input_ids (torch.Tensor): Input token IDs for the concatenated query and document.
            attention_mask (torch.Tensor): Attention mask for the input.
            memory_token_indices (torch.Tensor): Indices of the memory tokens in the input.
            rerank_token_index (torch.Tensor): Index of the reranking token.

        Returns:
            tuple: A tuple containing compressed_embeddings and rerank_score (if applicable).
        """
        outputs = self.compressor(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

        # Extract hidden states corresponding to memory tokens
        # memory_token_indices needs to be gathered correctly for batch
        batch_size = last_hidden_state.shape[0]
        compressed_embeddings = last_hidden_state[torch.arange(batch_size).unsqueeze(1), memory_token_indices]

        if not self.config.use_n_layers:
            compressed_embeddings = self.mapping(compressed_embeddings)

        rerank_score = None
        if self.config.add_reranking_head and rerank_token_index is not None:
            rerank_hidden_state = last_hidden_state[torch.arange(batch_size), rerank_token_index.squeeze()]
            rerank_score = self.rerank_head(rerank_hidden_state)

        return compressed_embeddings, rerank_score


class ScaleDownGenerator(PreTrainedModel):
    """
    ScaleDown Generator model.
    This model takes the compressed embeddings and the query and generates an answer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.generator = AutoModelForCausalLM.from_pretrained(config.name_or_path)

    def forward(self, input_ids, attention_mask, compressed_embeddings, labels=None):
        """
        Forward pass for the generator.
        """
        input_embeds = self.generator.get_input_embeddings()(input_ids)

        # Combine token embeddings with compressed document embeddings
        # This is a simplified example. A more robust implementation would handle placement.
        final_embeddings = torch.cat([compressed_embeddings, input_embeds], dim=1)

        # Adjust attention mask
        extended_attention_mask = torch.cat([
            torch.ones(compressed_embeddings.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask
        ], dim=1)


        outputs = self.generator(
            inputs_embeds=final_embeddings,
            attention_mask=extended_attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, input_ids, attention_mask, compressed_embeddings, **kwargs):
        input_embeds = self.generator.get_input_embeddings()(input_ids)
        final_embeddings = torch.cat([compressed_embeddings, input_embeds], dim=1)
        extended_attention_mask = torch.cat([
            torch.ones(compressed_embeddings.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask
        ], dim=1)

        return self.generator.generate(
            inputs_embeds=final_embeddings,
            attention_mask=extended_attention_mask,
            **kwargs
        )
