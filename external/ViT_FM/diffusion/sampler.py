
import torch
import tqdm
from scipy import integrate
import numpy as np
import time
from torch import nn

def Euler_Maruyama_sampler(model,
                           marginal_prob_std,
                           diffusion_coeff,
                           condition,
                           t_batch = None,
                           batch_size=64,
                           num_steps=128,
                           device='cuda',
                           dimension = (3,32,32),
                           eps=1e-3,
                           is_skip = False,
                           guidance_condition = None):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """

  if condition is not None:
    device = condition.device
  elif t_batch is not None:
    device = t_batch.device
  else:
    device = next(model.parameters()).device

  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, dimension[0], dimension[1], dimension[2], device=device) * marginal_prob_std(t).to(device)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x

  B,C,S,S = x.shape
  C_mask = guidance_condition.shape[1]
  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_time_step).to(device)

        #if guidance_condition is not None:
        #    guidance, dim = guidance_condition
        #    dim_in, dim_out = dim
        #    assert guidance.shape[0] == batch_size
        #    #x[:,dim_in:dim_out] = guidance
        spatial_mask = (torch.rand(S, S, device=device) < 0.1) 
        mask = spatial_mask.view(1, 1, S, S).expand(B, C, S, S)  # bool
        x[:, -C_mask:][mask] = guidance_condition[mask]

        if not is_skip:
          score = model(x, condition, batch_time_step, t_batch)
        else:
          denoised = model(x, condition, batch_time_step, t_batch)
          if guidance_condition is not None:
            denoised[:,dim_in:dim_out] = guidance
          std = marginal_prob_std(batch_time_step)
          score = (denoised - x)/std[:, None, None, None]**2
        mean_x = x + (g**2)[:, None, None, None] * score * step_size
        x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
        
  # Do not include any noise in the last sampling step.
  return mean_x

# import tqdm  # uncomment if you want a progress bar

@torch.no_grad()
def Euler_Maruyama_sampler_revised(
    model: nn.Module,
    marginal_prob_std,          # sigma(t): (B,) tensor -> (B,) tensor
    diffusion_coeff,            # g(t): (B,) tensor -> (B,) tensor
    condition=None,             # conditioning tensor or None
    t_batch=None,               # extra time-conditioning for your model, or None
    batch_size=64,
    num_steps=128,
    device=None,
    dimension=(3, 32, 32),
    eps=1e-3,
    is_skip=False,              # if True, model returns x0_hat; else, score
    guidance_condition=None,    # tuple: (guidance_tensor, (dim_in, dim_out)) or None
    guidance_strength=0.0,
):
    """
    Euler–Maruyama for reverse SDE sampling (VE-compatible).
    - Reverse drift: -g(t)^2 * score
    - Diffusion:     g(t) dW_bar
    """

    # Resolve device
    if device is None:
        if condition is not None:
            device = condition.device
        elif t_batch is not None:
            device = t_batch.device
        else:
            device = next(model.parameters()).device

    # Time grid (backward): t_0 = 1.0  ->  t_N = eps
    # We will use dt = t_{i+1} - t_i < 0
    time_steps = torch.linspace(1.0, eps, num_steps + 1, device=device)

    # Init from the perturbation kernel at t=1
    t0 = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, *dimension, device=device) * marginal_prob_std(t0).view(-1, 1, 1, 1)

    # Optional guidance
    #if guidance_condition is not None:
    #    guidance, (dim_in, dim_out) = guidance_condition
    #    assert guidance.shape[0] == batch_size, "Guidance batch size must match."

    B,C,S,S = x.shape
    if guidance_condition is not None:
      C_mask = guidance_condition.shape[1]
    mask = (torch.rand(B,C,S, S, device=device) < guidance_strength) 
    #mask = spatial_mask.view(1, 1, S, S).expand(B, C, S, S)  # bool
    if guidance_condition is not None:
      x[:, -C_mask:][mask] = guidance_condition[mask]

    for i in range(num_steps):
        t_cur  = time_steps[i]
        t_next = time_steps[i + 1]
        dt = t_next - t_cur  # negative

        t_vec = torch.ones(batch_size, device=device) * t_cur
        g = diffusion_coeff(t_vec).view(-1, 1, 1, 1)
        if is_skip:
            # model outputs x0_hat
            x0_hat = model(x, condition, t_vec, t_batch)
            if guidance_condition is not None and i < num_steps - 1:
                #spatial_mask = (torch.rand(S, S, device=device) < guidance_strength) 
                #mask = spatial_mask.view(1, 1, S, S).expand(B, C, S, S)  # bool
                x0_hat[:, -C_mask:][mask] = guidance_condition[mask]
            sigma_t = marginal_prob_std(t_vec).view(-1, 1, 1, 1)
            score = (x0_hat - x) / (sigma_t ** 2)
        else:
            # model outputs score directly
            score = model(x, condition, t_vec, t_batch)

        # Reverse SDE drift: -g^2 * score
        drift = -(g ** 2) * score
        x_mean = x + drift * dt  # dt < 0

        # Add noise on all but the last step
        if i < num_steps - 1:
            x = x_mean + g * torch.sqrt(-dt) * torch.randn_like(x)
        else:
            x = x_mean  # last step: no noise

    return x


## The error tolerance for the black-box ODE solver
@torch.no_grad()
def ode_sampler(model,
                marginal_prob_std,
                diffusion_coeff,
                condition,
                t_batch = None,
                batch_size=64,
                atol=1e-3,
                rtol=1e-3,
                device='cuda',
                z=None,
                dimension = (3,32,32),
                num_steps = None,
                is_skip = True,
                guidance_condition = None,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, dimension[0], dimension[1], dimension[2], device=device) * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z

  shape = init_x.shape

  #print(shape)
  #time.sleep(1000)

  def score_eval_wrapper(x, t_vec):
    """A wrapper of the score-based model for use by the ODE solver."""
    x = torch.tensor(x, device=device, dtype=torch.float32).reshape(shape)
    t_vec = torch.tensor(t_vec, device=device, dtype=torch.float32).reshape((x.shape[0], ))
    #with torch.no_grad():
    #  score = score_model(sample, condition, time_steps, t_bacth)

    x0_hat = model(x, condition, t_vec, t_batch)
    sigma_t = marginal_prob_std(t_vec).view(-1, 1, 1, 1)
    score = (x0_hat - x) / (sigma_t ** 2)

    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='BDF')
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x
