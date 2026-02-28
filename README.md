# PsychologicallyConditioned-Mobility

Overview
This project implements multiple deep learning architectures for
multi-day human mobility prediction by integrating latent psychological
needs into sequential modeling frameworks. The models combine behavioral
sequence learning with attention mechanisms to improve predictive
accuracy and interpretability.

Implemented Models

1.  Bidirectional LSTM with Attention
    -   Captures forward and backward temporal dependencies
    -   Learns long-range behavioral patterns
    -   Uses attention to identify influential time steps
2.  Temporal Convolutional Network (TCN) with Attention
    -   Uses stacked 1D convolutional layers with residual connections
    -   Efficient parallel computation for long sequences
    -   Attention highlights psychologically salient time intervals
3.  Transformer Model
    -   Uses self-attention to capture global temporal dependencies
    -   Fully parallelizable and scalable
    -   Provides interpretable attention weights
4.  Need-Aware Attention Model
    -   Integrates three latent psychological needs
    -   Needs represented as learned embeddings or latent vectors
    -   Attention conditioned on motivational states
    -   Enables psychologically grounded prediction

Prediction Pipeline

1.  Historical mobility sequences are embedded.
2.  Latent psychological need vectors are fused with mobility features.
3.  Sequential architecture (BiLSTM, TCN, or Transformer) extracts
    temporal features.
4.  Attention mechanism assigns importance weights to time steps.
5.  Context vector is computed using weighted temporal features.
6.  Final layer predicts next activity or mobility outcome.

Evaluation Metrics

-   Accuracy
-   Mean Absolute Error (MAE)
-   Mean Squared Error (MSE)
-   Training and Testing Loss Curves
-   KDE-based accuracy visualization

Technologies Used

-   Python
-   PyTorch
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn

Applications

-   Smart transportation systems
-   Travel demand forecasting
-   Urban planning
-   Behavior-aware AI systems
-   Human-centered predictive modeling

