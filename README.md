
````
# Transformer From Scratch: A Hands-On Implementation

This repository contains a PyTorch implementation of the Transformer model, based on the seminal paper "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)". This project aims to demystify the internal workings of Transformers by building them from the ground up, providing a clear and well-commented codebase.

##  Project Highlights

* **Modular Implementation:** Each core component of the Transformer (Self-Attention, Encoder, Decoder, Transformer Block) is implemented as a separate `nn.Module`.
* **Detailed Explanation:** The code is designed to be easily understandable, with inline comments and clear variable names.
* **Key Learnings:** This project served as a hands-on learning experience for understanding the intricate details of Transformer architecture.
* **Simple Data Pipeline:** Includes a basic `TextDataset` and `SimpleTokenizer` for demonstration, making it easy to get started with small textual datasets.

##  Learnings from Implementing Transformers

Implementing the Transformer from scratch provided deep insights into its revolutionary design. Here are some key takeaways:

### 1. The Power of Self-Attention

The core innovation of the Transformer is the self-attention mechanism. Unlike recurrent neural networks (RNNs) that process sequences sequentially, self-attention allows the model to weigh the importance of different words in an input sequence when processing each word. This enables parallelization and captures long-range dependencies more effectively.

**Code Snippet: `SelfAttention` Module**

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embed size should be divisible by heads"

        # Linear layers for Query, Key, Value transformations
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Output linear layer after concatenating heads
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.softmax = nn.Softmax(dim=3) # Softmax applied over the key dimension (dim=3)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embeddings into 'heads' pieces
        # Shape: (N, seq_len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Apply linear transformations to queries, keys, values for each head
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calculate energy (dot product of queries and keys)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply mask to energy (for padding or future tokens)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) # Set masked values to a very small number

        # Apply softmax to get attention probabilities
        attention = self.softmax(energy / (self.embed_size**(1/2))) # Scaled dot-product attention

        # Multiply attention weights with values
        # out shape: (N, query_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        # Concatenate heads and pass through final linear layer
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
````

### 2\. Positional Encodings

Since Transformers process all words in parallel without inherent sequential understanding, **positional encodings** are crucial. These are added to the word embeddings to inject information about the relative or absolute position of tokens in the sequence.

**Code Snippet: Positional Embedding in `Encoder`**

```python
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length):
        super(Encoder,self).__init__()
        self.embed_size=embed_size
        self.device=device
        self.word_embedding=nn.Embedding(src_vocab_size,embed_size)
        self.position_embedding=nn.Embedding(max_length,embed_size) # Positional embedding layer
        self.layers=nn.ModuleList([
            Transformer(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion
            )
            for _ in range(num_layers)
        ])
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask):
        N,seq_length=x.shape
        # Generate positions for the input sequence
        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        # Combine word and positional embeddings
        out=self.dropout(self.word_embedding(x)+self.position_embedding(positions))
        for layer in self.layers:
            out=layer(out,out,out,mask) # Self-attention in encoder
        return out
```

### 3\. Encoder-Decoder Architecture

The Transformer follows an encoder-decoder structure.

  * **Encoder:** Processes the input sequence and produces a rich representation (contextualized embeddings).
  * **Decoder:** Uses this encoded information along with its own masked self-attention to generate the output sequence one token at a time. The decoder's "cross-attention" mechanism attends to the encoder's output.

**Code Snippet: `DecoderBlock` and `Decoder` Structure**

```python
class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock,self).__init__()
        # Masked Self-Attention for decoder input
        self.attention=SelfAttention(embed_size,heads)
        self.norm=nn.LayerNorm(embed_size) # Normalization after attention
        # Transformer Block includes cross-attention and feed-forward
        self.transformer_block=Transformer(
            embed_size,heads,dropout,forward_expansion)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,value,key,src_mask,trg_mask):
        # 1. Masked Self-Attention on decoder input (x, x, x with trg_mask)
        attention=self.attention(x,x,x,trg_mask)
        query=self.dropout(self.norm(attention+x)) # Add & Norm

        # 2. Encoder-Decoder Attention (Cross-Attention)
        # Queries come from decoder (query), Keys/Values from encoder output (value, key)
        out=self.transformer_block(value,key,query,src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder,self).__init__()
        self.device=device
        self.word_embedding=nn.Embedding(trg_vocab_size,embed_size)
        self.position_embedding=nn.Embedding(max_length,embed_size)
        self.layers=nn.ModuleList(
            [DecoderBlock(embed_size,heads,forward_expansion,dropout,device)
            for _ in range(num_layers)]
        )
        self.fc_out=nn.Linear(embed_size,trg_vocab_size) # Final linear layer for vocabulary prediction
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,enc_out,src_mask,trg_mask):
        N,seq_length=x.shape
        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        x=self.dropout((self.word_embedding(x)+self.position_embedding(positions)))

        for layer in self.layers:
            # x is the decoder's current input, enc_out provides key and value from encoder
            x=layer(x,enc_out,enc_out,src_mask,trg_mask)

        out=self.fc_out(x) # Project to vocabulary size
        return out
```

### 4\. Layer Normalization and Residual Connections

Crucial for stable training of deep networks, **Layer Normalization** is applied after the attention and feed-forward sub-layers. **Residual connections** (or skip connections) are used around each sub-layer, followed by layer normalization. This helps prevent the vanishing gradient problem and allows for deeper models.

**Code Snippet: Residual Connections and LayerNorm in `Transformer` Block**

```python
class Transformer(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super(Transformer,self).__init__()
        self.attention=SelfAttention(embed_size,heads)
        self.norm1=nn.LayerNorm(embed_size) # LayerNorm for the first sub-layer
        self.norm2=nn.LayerNorm(embed_size) # LayerNorm for the second sub-layer

        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size,forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self,value,key,query,mask):
        # First sub-layer: Multi-Head Attention
        attention=self.attention(value,key,query,mask)
        # Add & Norm: attention + query (residual connection), then LayerNorm, then Dropout
        x=self.dropout(self.norm1(attention+query))

        # Second sub-layer: Feed-Forward Network
        forward=self.feed_forward(x)
        # Add & Norm: forward + x (residual connection), then LayerNorm, then Dropout
        out=self.dropout(self.norm2(forward+x))
        return out
```

### 5\. Masking Strategies

Masking is vital to prevent the model from cheating:

  * **Source Mask (`make_src_mask`):** Prevents attention to padding tokens in the input sequence, ensuring they don't influence attention scores.
  * **Target Mask (`make_trg_mask`):** Applied in the decoder's self-attention to prevent attending to future tokens in the target sequence, enforcing an autoregressive generation process during training.

**Code Snippet: Masking Functions in `TransformerModel`**

```python
class TransformerModel(nn.Module):
    # ... (init method) ...

    def make_src_mask(self,src):
      # Creates a boolean mask: True where src != pad_idx, False otherwise
      # Unsqueezing to match expected dimensions for broadcasting in attention
      src_mask=(src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
      return src_mask.to(self.device)

    def make_trg_mask(self,trg):
      N,trg_len=trg.shape
      # Creates a lower triangular matrix of ones (look-ahead mask)
      trg_mask=torch.tril(torch.ones((trg_len,trg_len))).expand(
          N,1,trg_len,trg_len # Expand for batch and head dimensions
      )
      return trg_mask.to(self.device)

    def forward(self,src,trg):
      # Generate masks based on input and target sequences
      src_mask=self.make_src_mask(src)
      trg_mask=self.make_trg_mask(trg)
      # Pass masks to encoder and decoder
      enc_src=self.encoder(src,src_mask)
      out=self.decoder(trg,enc_src,src_mask,trg_mask)
      return out
```

##  Getting Started

### Prerequisites

  * Python 3.x
  * PyTorch (`torch`)

### Installation

Clone the repository:

```bash
git clone[(https://github.com/SharmajiKabetaDevesh/transformers_with_torch.git)](https://github.com/SharmajiKabetaDevesh/transformers_with_torch.git)
cd transformers_with_torch
```

### Running the Code

The provided code includes a simple example of training the Transformer model on a very small, synthetic dataset.

1.  **Ensure PyTorch is installed:**

    ```bash
    pip install torch
    ```

2.  **Run the script:**

    ```bash
    python your_main_script_file.py # (Assuming you put the classes and training loop in a single file)
    ```

    You will see output similar to:

    ```
    Starting training...
    Epoch [1/100], Loss: 1.2345
    Epoch [2/100], Loss: 1.1234
    ...
    ```

## 📚 Further Reading & References

Understanding Transformers deeply requires grasping several core concepts. Here are some excellent resources that helped in this project:

  * **The Original Paper:**

      * [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani et al. (2017)

  * **Visual Guides & Explanations (Highly Recommended for Beginners):**

      * **"The Illustrated Transformer" by Jay Alammar:** A phenomenal visual explanation of the Transformer architecture.
          * [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
      * **Medium Blogs:** Many excellent articles break down Transformers step-by-step. Searching for "Transformer explained medium" will yield many results. Here are a couple of examples that are usually well-regarded:
          * [A Step-by-Step Guide to the Transformer Model](https://www.google.com/search?q=https://towardsdatascience.com/a-step-by-step-guide-to-the-transformer-model-f7f5029a1b1a) (Towards Data Science)
          * [Attention Is All You Need: The Transformer explained](https://www.google.com/search?q=https://medium.com/%40lsh.sharma/attention-is-all-you-need-the-transformer-explained-823908f51a70) (LSh.Sharma)

  * **PyTorch Documentation:**

      * [PyTorch `nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
      * [PyTorch `nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
      * [PyTorch `nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
      * [PyTorch `nn.LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

##  Contributing

Feel free to open issues or submit pull requests if you find bugs, have suggestions for improvements, or want to add more features (e.g., Beam Search for inference, support for larger datasets).

## 📄 License

This project is open-sourced under the MIT License. 

```
```
