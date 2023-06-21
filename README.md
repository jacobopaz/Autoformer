# Autoformer Encoder

Autoformer encoder block introduced in the paper: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting".

It modifies the self attention mechanism by changing from time domain to frequency domain when calculating the attention weights. Then it aggregates these weights by time delay. This new implementation enables the mechanism to focus on period-based dependencies.
