import torch
import torch.nn as nn
from torchdiffeq import odeint

# Check if mps is available for Apple Silicon and set device accordingly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.device(device)

class ODEFunc(nn.Module):
    """
    The 'Dynamics' function. 
    Models how the patient's hidden state evolves naturally over time 
    (e.g., how a drug concentration decays or physiology stabilizes).
    """
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, t, h):
        return self.net(h)

class ODESolver(nn.Module):
    """
    A simple Euler solver to integrate the ODE. 
    (In production, use torchdiffeq.odeint for adaptive step solvers like Runge-Kutta)
    """
    # Use torchdiffeq and replace this Euler solver for better accuracy.
    
    def __init__(self, func):
        super(ODESolver, self).__init__()
        self.func = func

    def forward(self, h_start, t_start, t_end, steps=10):
        """
        dt = (t_end - t_start) / steps
        h = h_start
        t = t_start
        
        for _ in range(steps):
            dh = self.func(t, h)
            h = h + dh * dt
            t = t + dt
        return h
        """
        t = torch.linspace(t_start, t_end, steps).to(h_start.device)
        h = odeint(self.func, h_start, t, method="rk4")
        return h[-1]

class ClinicalODERNN(nn.Module):
    """
    Stabilized ODE-RNN for clinical trajectory encoding.

    Key stabilization features:
    1. Time scaling to [0, 1] for solver stability
    2. LayerNorm on output to prevent gradient explosion into projection layer
    3. Augmented state: delta-time passed explicitly to GRU update gate
    4. Adaptive step solver (dopri5) for irregular time series
    """
    def __init__(self, input_dim, hidden_dim, time_scale=72.0,
                 solver_method='dopri5', rtol=1e-3, atol=1e-4):
        super(ClinicalODERNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_scale = time_scale  # Max expected time value for normalization

        # 1. The ODE Dynamics (State Decay/Evolution)
        self.ode_func = ODEFunc(hidden_dim)

        # 2. Solver settings - adaptive step size for irregular clinical data
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol

        # 3. Augmented input: GRU receives (value, delta_time) to learn rate-awareness
        self.gru_update = nn.GRUCell(input_dim + 1, hidden_dim)

        # 4. LayerNorm - critical for stabilizing gradients before projection layer
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, time_points, values):
        """
        Args:
            time_points: (Batch, Seq) - e.g., [0.0, 24.0, 72.0]
            values: (Batch, Seq, Feature_Dim) - e.g., [[24.], [20.], [12.]]
        Returns:
            Normalized trajectory embeddings: (Batch, Seq, hidden_dim)
        """
        batch_size, seq_len, _ = values.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=values.device)

        # Scale time globally to [0, 1] for solver stability
        times_scaled = time_points / self.time_scale

        outputs = []

        for i in range(seq_len):
            # A. EVOLVE (Solve ODE from t_prev to t_current)
            if i > 0:
                # Use batch mean for time points (approximation for batch processing)
                t_prev = times_scaled[:, i-1].mean()
                t_curr = times_scaled[:, i].mean()

                # Only integrate if time has advanced sufficiently (avoid ODE solver underflow)
                dt_val = (t_curr - t_prev).item() if hasattr(t_curr - t_prev, 'item') else float(t_curr - t_prev)
                min_dt = 1e-4  # Minimum time gap to avoid numerical issues

                if dt_val > min_dt:
                    try:
                        t_span = torch.tensor([t_prev.item(), t_curr.item()], device=h.device, dtype=h.dtype)
                        h = odeint(
                            self.ode_func, h, t_span,
                            method=self.solver_method,
                            rtol=self.rtol,
                            atol=self.atol
                        )[1]  # Take final state
                    except AssertionError:
                        # ODE solver underflow - skip integration, just use GRU update
                        pass

            # B. UPDATE with augmented features (value + delta_time)
            # Calculate local dt for the GRU (helps model learn rate of change)
            # Note: dt is NOT normalized - raw time gap is informative for learning
            if i > 0:
                dt = (time_points[:, i] - time_points[:, i-1]).unsqueeze(-1)
            else:
                dt = torch.zeros(batch_size, 1, device=values.device)

            # Concatenate value features with raw time delta
            x_t = values[:, i, :]
            input_feats = torch.cat([x_t, dt], dim=-1)
            h = self.gru_update(input_feats, h)

            outputs.append(h)

        # Stack and normalize - fixes gradient explosion entering projection layer
        rnn_out = torch.stack(outputs, dim=1)
        return self.norm(rnn_out)

# Example Usage
if __name__ == "__main__":
    # Input: (Batch=1, Seq=3, Dim=1) -> 3 measurements of Anion Gap
    values = torch.tensor([[[24.], [20.], [12.]]])
    # Time: 0h -> 24h -> 72h
    times = torch.tensor([[0., 24., 72.]])

    model = ClinicalODERNN(input_dim=1, hidden_dim=16, time_scale=72.0)
    trajectory_embedding = model(times, values)

    print(f"Output Shape: {trajectory_embedding.shape}")
    # Shape: [1, 3, 16]
    # This vector now encodes the *dynamics* of the 72h gap, not just the values.

    # Verify LayerNorm output stats
    print(f"Output mean: {trajectory_embedding.mean():.4f}")
    print(f"Output std: {trajectory_embedding.std():.4f}")