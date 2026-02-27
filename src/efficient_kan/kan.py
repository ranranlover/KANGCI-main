import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        # output = output.reshape(*original_shape[:-1], self.out_features)




        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

   #  def GC(self):
   #      GC = self.layers[0].base_weight
   #      W = GC.unsqueeze(2)
   #      weight_norm = torch.norm(W, dim=(0, 2))
   #      return weight_norm
   #  #4: 1 2 3 4 5  3:
   #  def GC_weighted_by_sensitivity(self, x_sample, method='grad_input', ig_steps=20, smooth_n=3, eps=1e-12):
   #      """
   #      Weight base_weight columns by a per-feature sensitivity and then compute norm.
   #      - x_sample: (1, T, P) representative input
   #      - method: 'grad_input' | 'ig' | 'smooth_ig'
   #      - ig_steps: steps for IG if used
   #      - smooth_n: number of noise samples for smooth IG
   #      Returns: (P,) tensor on device
   #      """
   #      device = next(self.parameters()).device
   #      x = x_sample.clone().detach().to(device)
   #      if x.dim() > 1 and x.shape[0] > 1:
   #          x = x[0:1]
   #      in_features = self.layers[0].in_features
   #
   #      # compute sensitivity vector (length in_features)
   #      if method == 'grad_input':
   #          xg = x.clone().detach().requires_grad_(True)
   #          out = self.forward(xg)
   #          out_sum = out.sum()
   #          grads = torch.autograd.grad(out_sum, xg, retain_graph=False, create_graph=False)[0]  # (1, ..., F)
   #          grads_flat = grads.reshape(-1, in_features).mean(dim=0)  # (F,)
   #          inputs_flat = x.reshape(-1, in_features).mean(dim=0)  # (F,)
   #          sens = (grads_flat * inputs_flat).abs()  # Grad × Input
   #      elif method == 'ig' or method == 'smooth_ig':
   #          # reuse GC_integrated_v1 or GC_smooth_ig-like logic
   #          if method == 'ig':
   #              sens = self.GC_integrated_v1(x, steps=ig_steps)
   #          else:
   #              sens = self.GC_smooth_ig(x, steps=ig_steps, n_samples=smooth_n)
   #      else:
   #          raise ValueError("method must be 'grad_input'|'ig'|'smooth_ig'")
   #
   #      sens = sens.to(device)
   #      # normalize to [0,1]
   #      sens = sens / (sens.max().clamp(min=eps))
   #
   #      # now weight base_weight columns: base_weight shape (out_features, in_features)
   #      W = self.layers[0].base_weight  # (out, in)
   #      if W.dim() == 1:
   #          # treat as (1, in)
   #          W_mat = W.unsqueeze(0)
   #      else:
   #          W_mat = W
   #
   #      # scale each column by (1 + sens) or sens directly (choose small boost to avoid zeroing)
   #      scaled_W = W_mat * (1.0 + sens.unsqueeze(0))
   #      # compute norm over out_features and any extra dims similar to original GC
   #      weight_norm = torch.norm(scaled_W, dim=0)  # (in_features,)
   #      return weight_norm.detach()
   #
   #  def GC_full_jacobian(self, x_sample):
   #      """
   #      True Jacobian-based GC: compute || dy_j / dx_i || across full network.
   #      Returns: (P,) GC vector.
   #      """
   #      x = x_sample.clone().detach().requires_grad_(True)
   #      y = self.forward(x)  # (1, T, out)
   #      y_sum = y.sum(dim=1)  # aggregate over time if necessary
   #      J = []
   #      for j in range(y_sum.shape[-1]):
   #          grad_j = torch.autograd.grad(y_sum[0, j], x, retain_graph=True)[0]  # dy_j/dx
   #          J.append(grad_j.abs().mean(dim=(0, 1)))  # (in_features,)
   #      J = torch.stack(J, dim=0)
   #      GC_vector = J.norm(dim=0)  # aggregate across outputs
   #      return GC_vector.detach()
   #
   #  # 放在模型类里 (e.g. class KAN(...): )
   #  def GC_full_jacobian_plus(self, x_sample, n_samples=8, noise_sigma=0.02, clip_val=10.0, time_reduce='mean', eps=1e-12):
   #      """
   #      Stable Jacobian-based GC:
   #      - SmoothGrad-like averaging (n_samples) with Gaussian noise (noise_sigma).
   #      - For each perturbation compute dy/dx, aggregate abs-grad over time and samples,
   #        then aggregate across outputs and return a (P,) GC vector (per input feature).
   #      Args:
   #          x_sample: representative input tensor, shape (1, T, P) or (B, T, P)
   #          n_samples: number of noise samples for averaging (>=1)
   #          noise_sigma: std dev for additive Gaussian noise (small e.g. 0.01-0.05)
   #          clip_val: clip per-dimension grads to [-clip_val, clip_val]
   #          time_reduce: 'mean'|'max' reduction over time axis when collapsing grads
   #      Returns:
   #          GC_vector (P,) on same device as model
   #      """
   #      device = next(self.parameters()).device
   #      x_base = x_sample.clone().detach().to(device)
   #      # ensure batch dim = 1 for attribution
   #      if x_base.dim() == 2:
   #          # (T, P) -> (1, T, P)
   #          x_base = x_base.unsqueeze(0)
   #      if x_base.shape[0] > 1:
   #          x = x_base[0:1]
   #      else:
   #          x = x_base
   #
   #      in_features = x.shape[-1]
   #
   #      # accumulate per-sample abs grads reduced over time -> shape (in_features,)
   #      sens_accum = x.new_zeros(in_features)
   #      for _ in range(max(1, n_samples)):
   #          # add small noise for smoothness
   #          x_pert = x + noise_sigma * torch.randn_like(x)
   #          x_pert = x_pert.clone().detach().requires_grad_(True)
   #
   #          y = self.forward(x_pert)  # expected shape: (1, T, out) or (1, out)
   #          # aggregate outputs if time dimension present
   #          if y.dim() == 3:
   #              # (1, T, out) -> we want per-output scalars to backprop separately
   #              # form y_sum over time for each output
   #              y_sum = y.sum(dim=1)  # (1, out)
   #          else:
   #              y_sum = y  # assume (1, out) or (1, out_dim)
   #              if y_sum.dim() == 1:
   #                  y_sum = y_sum.unsqueeze(0)
   #
   #          # for stability compute grads per output and aggregate absolute value
   #          # accumulate per-output grads then collapse output axis via norm
   #          # compute gradient of each output w.r.t x
   #          # We'll compute sum of outputs to get gradient that captures combined sensitivity,
   #          # but also compute per-output grads if needed to be more precise.
   #          # Here compute grad of sum(all outputs) -> a surrogate for overall sensitivity.
   #          total_out = y_sum.sum()
   #          grad = torch.autograd.grad(total_out, x_pert, create_graph=False, retain_graph=False)[0]  # (1, T, P)
   #
   #          # reduce time dimension
   #          if time_reduce == 'mean':
   #              grad_reduced = grad.abs().mean(dim=1).view(-1)  # (P,)
   #          elif time_reduce == 'max':
   #              grad_reduced = grad.abs().max(dim=1)[0].view(-1)
   #          else:
   #              grad_reduced = grad.abs().mean(dim=1).view(-1)
   #
   #          # clip and accumulate
   #          if clip_val is not None:
   #              grad_reduced = torch.clamp(grad_reduced, -clip_val, clip_val).abs()
   #          sens_accum += grad_reduced.detach()
   #
   #      sens = sens_accum / float(max(1, n_samples))
   #
   #      # normalize sens to [0,1]
   #      sens = sens / (sens.max().clamp(min=eps))
   #
   #      # Now combine with first-layer weights similar to original GC: base_weight shape (out, in)
   #      W = self.layers[0].base_weight  # (out_features, in_features) or (in_features,) if weird
   #      if W.dim() == 1:
   #          W_mat = W.unsqueeze(0)
   #      else:
   #          W_mat = W
   #
   #      # mild scaling: avoid zeroing by using (1 + alpha*sens) with alpha small
   #      alpha = 0.5
   #      scaled_W = W_mat * (1.0 + alpha * sens.unsqueeze(0))
   #
   #      # compute norm over output dim -> (in_features,)
   #      weight_norm = torch.norm(scaled_W, dim=0)
   #
   #      # final normalization (optional): scale to [0,1] for comparability
   #      weight_norm = weight_norm / (weight_norm.max().clamp(min=eps))
   #
   #      return weight_norm.detach()
   #
   #  import torch
   # #  yakebi  haibucuo
   #  def GC_jjj(self):
   #      """
   #      Jacobian-based GC vector (returns shape (P,)).
   #      - If model has an example input (self.example_input or self.last_input), compute
   #        per-output Jacobian norm: for each output j, norm( d y_j / d x ).
   #      - Otherwise fall back to the original weight-norm behavior for compatibility.
   #      """
   #      device = next(self.parameters()).device
   #
   #      # Try to find a representative input automatically
   #      x_sample = None
   #      for attr in ("example_input", "last_input", "gc_input"):
   #          if hasattr(self, attr):
   #              x_sample = getattr(self, attr)
   #              break
   #
   #      if x_sample is None:
   #          # Fall back to original behavior (compatibility)
   #          GC = self.layers[0].base_weight  # original tensor
   #          W = GC.unsqueeze(2)
   #          weight_norm = torch.norm(W, dim=(0, 2))  # keep original output shape
   #          return weight_norm.to(device)
   #
   #      # Ensure x_sample is a single example on the same device and requires grad
   #      x = x_sample.clone().detach().to(device)
   #      # If batch given, use first sample (so result corresponds to single sample jacobian)
   #      if x.dim() > 1 and x.shape[0] > 1:
   #          x = x[0:1]
   #      x.requires_grad_(True)
   #
   #      # Forward pass: expect y shape (1, P) or (P,)
   #      y = self.forward(x)
   #      y = y.view(-1)  # make (P,)
   #      P = y.shape[0]
   #
   #      norms = []
   #      for j in range(P):
   #          # clear any prev grads on x to avoid accumulation
   #          if x.grad is not None:
   #              x.grad.detach_()
   #              x.grad.zero_()
   #
   #          grad_x = torch.autograd.grad(
   #              outputs=y[j],
   #              inputs=x,
   #              retain_graph=True,
   #              create_graph=True,
   #              allow_unused=True
   #          )[0]  # same shape as x or None
   #
   #          if grad_x is None:
   #              norms.append(torch.tensor(0.0, device=device))
   #          else:
   #              norms.append(torch.norm(grad_x.view(-1)))
   #
   #      per_output_norm = torch.stack(norms, dim=0)  # (P,)
   #      return per_output_norm
   #
   #
   #
   #
   #  def GC_jacobian_exact(self, x_sample=None, seq_len=10):
   #      """
   #      精确版：对 y_last = y_pred[:, -1, :] （shape (batch, P_out)）计算
   #      相对于输入 x (1, T_in, P_in) 的雅可比，并返回对输入特征维度（P_in）上的范数：
   #          weight_norm[j] = || jacobian wrt input_feature j ||_2  （时间维度上合并）
   #      注意：精确但可能非常慢 / 占内存，适合 P_in, seq_len 小的情况。
   #      """
   #      device = next(self.parameters()).device
   #
   #      # ---- 推断 input_dim ----
   #      input_dim = None
   #      if hasattr(self, "input_dim") and self.input_dim is not None:
   #          input_dim = int(self.input_dim)
   #      else:
   #          bw = getattr(self.layers[0], "base_weight", None)
   #          if bw is not None:
   #              input_dim = int(bw.shape[1])
   #          else:
   #              w = getattr(self.layers[0], "weight", None)
   #              if w is not None:
   #                  input_dim = int(w.shape[1])
   #      if input_dim is None:
   #          raise AttributeError(
   #              "无法推断输入维度 (input_dim)。请设置 self.input_dim 或确保 layers[0] 有 base_weight/weight。"
   #          )
   #
   #      # ---- 构建/调整 x_sample ----
   #      if x_sample is None:
   #          x_sample = torch.randn(1, seq_len, input_dim, device=device)
   #      else:
   #          x_sample = x_sample.to(device)
   #      x_sample.requires_grad_(True)
   #
   #      # ---- 前向 ----
   #      y_pred = self.forward(x_sample)  # (batch, T_out, P_out)
   #      y_last = y_pred[:, -1, :]  # (batch, P_out)
   #
   #      # 计算完整的 Jacobian: dy_last/dx_sample -> shape (batch, P_out, batch, T_in, P_in)
   #      # 为简化, 先把 batch 合并（通常 batch==1）
   #      # 使用 torch.autograd.functional.jacobian （注意内存）
   #      def forward_to_ylast(inp):
   #          out = self.forward(inp)[:, -1, :]  # (batch, P_out)
   #          # 为 autograd.functional.jacobian 要求返回同样的 shape
   #          return out
   #
   #      # jac: shape (inp_shape..., out_shape...)
   #      jac = torch.autograd.functional.jacobian(
   #          forward_to_ylast, x_sample, create_graph=False
   #      )  # large tensor: (batch, P_out, batch, T_in, P_in)
   #
   #      # 由于我们通常 batch == 1, 拆成 (P_out, T_in, P_in)
   #      # 若 batch>1, 需按需聚合。这里按 batch 维相等的假设处理：
   #      if jac.ndim == 5:
   #          # case: (batch, P_out, batch, T_in, P_in)
   #          # 合并 batch dims (assume single batch)
   #          jac = jac.squeeze(0).squeeze(1) if jac.shape[0] == jac.shape[2] == 1 else jac
   #      # 最终期望得到 (P_out, T_in, P_in) or (T_in, P_in) if P_out summed
   #
   #      # 计算对输入 feature 的范数：先对 P_out/time 合并，再保留 input feature 维
   #      # 如果 jac has P_out dim:
   #      if jac.ndim == 3:
   #          # (P_out, T_in, P_in)
   #          # 对输出维(P_out)与时间维(T_in)求平方和后开方 -> (P_in,)
   #          weight_norm = torch.norm(jac.reshape(-1, jac.shape[-1]), dim=0)
   #      elif jac.ndim == 2:
   #          # (T_in, P_in)
   #          weight_norm = torch.norm(jac, dim=0)
   #      else:
   #          # fallback: compute grads via autograd as in original
   #          grad_all = torch.autograd.grad(y_last.sum(), x_sample, create_graph=False)[0]
   #          weight_norm = torch.norm(grad_all, dim=(0, 1))
   #
   #      return weight_norm
   #
   #  def GC_path_integral(self, x_sample=None, baseline=None, steps=50):
   #      """
   #      基于路径积分（Integrated Gradients）的 KAN 因果强度提取方法
   #      返回形式与原 GC() 一致：每个输入维度的平均路径归因强度。
   #
   #      Args:
   #          x_sample: 一个样本（或多个样本）tensor, shape = [N, in_features]
   #          baseline: 基线输入（默认取零向量或均值）
   #          steps: 路径积分步数 (越大越平滑)
   #      """
   #      device = next(self.parameters()).device
   #
   #      # Step 1. 准备输入样本
   #      if x_sample is None:
   #          raise ValueError("必须提供输入样本 x_sample 来计算路径积分。")
   #
   #      x_sample = x_sample.to(device)
   #      if x_sample.ndim == 1:
   #          x_sample = x_sample.unsqueeze(0)
   #
   #      # Step 2. 定义基线（默认全零）
   #      if baseline is None:
   #          baseline = torch.zeros_like(x_sample).to(device)
   #
   #      # Step 3. 生成路径点
   #      alphas = torch.linspace(0, 1, steps).to(device)
   #      integrated_grad = torch.zeros_like(x_sample).to(device)
   #
   #      for alpha in alphas:
   #          x_alpha = baseline + alpha * (x_sample - baseline)
   #          x_alpha.requires_grad_(True)
   #          output = self.forward(x_alpha)
   #          # 对所有输出求和（得到总体影响）
   #          output_sum = output.sum()
   #          grads = torch.autograd.grad(output_sum, x_alpha, retain_graph=False)[0]
   #          integrated_grad += grads
   #
   #      # Step 4. 平均梯度并乘以输入变化（积分近似）
   #      avg_grad = integrated_grad / steps
   #      ig = (x_sample - baseline) * avg_grad  # Integrated Gradients
   #
   #      # Step 5. 求绝对值平均作为强度（保持输出格式一致）
   #      causal_strength = ig.abs().mean(dim=0)  # [in_features]
   #      return causal_strength.detach()
   #  # 3  buhao   4  hao
   #  def GC_integrated_v1(self, x_sample, target_idx=None, baseline=None, steps=50):
   #      """
   #      Integrated-Gradiens style path-integral attribution returning a vector of length in_features.
   #      - x_sample: tensor with shape (..., seq_len, in_features) OR (batch, in_features) etc.
   #                  In your training loop you pass `input_seq` of shape [1, T, P] which is supported.
   #      - target_idx: kept for API-compatibility (not used because each KAN model outputs scalar per sample).
   #      - baseline: baseline tensor same shape as x_flat (defaults to zeros).
   #      - steps: number of Riemann steps for the integral (tradeoff speed/accuracy).
   #      Returns:
   #          causal_strength: tensor shape (in_features,) same device as the model.
   #      """
   #      device = next(self.parameters()).device
   #      # --- prepare x_flat: shape (N, in_features) where N = prod(all leading dims except last) ---
   #      x = x_sample.to(device)
   #      # ensure last dim equals in_features
   #      if x.dim() < 1 or x.size(-1) != self.layers[0].in_features:
   #          raise ValueError(f"x_sample last dim must equal in_features={self.layers[0].in_features}")
   #
   #      original_shape = x.shape  # keep for possible reshape (not strictly needed here)
   #      in_features = x.size(-1)
   #      x_flat = x.reshape(-1, in_features)  # (N, in_features)
   #      N = x_flat.size(0)
   #
   #      # baseline
   #      if baseline is None:
   #          baseline = torch.zeros_like(x_flat, device=device)
   #      else:
   #          baseline = baseline.to(device).reshape(-1, in_features)
   #          if baseline.size(0) != x_flat.size(0):
   #              # allow single baseline broadcast
   #              baseline = baseline.expand_as(x_flat)
   #
   #      # allocate integrated grads
   #      integrated_grad = torch.zeros_like(x_flat, device=device)
   #
   #      # path integral (Riemann sum). We compute grads of summed outputs to aggregate influence across all samples.
   #      # Vectorized alpha generation but loop over alphas to save memory.
   #      alphas = torch.linspace(0.0, 1.0, steps, device=device)
   #      for alpha in alphas:
   #          x_alpha = baseline + alpha * (x_flat - baseline)
   #          x_alpha.requires_grad_(True)
   #
   #          # forward: KAN.forward accepts (..., in_features) shapes; x_alpha is (N, in_features)
   #          out = self.forward(x_alpha)  # shape (N, out_features) — for your networks out_features==1
   #          # Sum all outputs so that gradients reflect aggregated effect across all time/batch samples
   #          out_sum = out.sum()
   #          grads = torch.autograd.grad(out_sum, x_alpha, retain_graph=False, create_graph=False)[0]
   #          # grads shape (N, in_features)
   #          integrated_grad += grads
   #
   #          # free graph
   #          x_alpha.requires_grad_(False)
   #
   #      avg_grad = integrated_grad / float(steps)  # average gradient along path
   #      # integrated gradients approximation
   #      ig = (x_flat - baseline) * avg_grad  # shape (N, in_features)
   #
   #      # aggregate to per-feature causal strength: match original GC which returned one value per input feature
   #      # we take absolute mean across samples (you can change to mean or L2 if desired)
   #      causal_strength = ig.abs().mean(dim=0)  # (in_features,)
   #
   #      return causal_strength.detach()
   #
   #  # 1️⃣ 稳定版 Integrated Gradients
   #  def GC_integrated_stable(self, x_sample, baseline=None, steps=50, n_samples=8,
   #                           noise_sigma=0.02, clip_val=5.0, eps=1e-12):
   #      device = next(self.parameters()).device
   #      x = x_sample.to(device)
   #      if x.dim() == 3:
   #          x_flat = x.view(-1, x.size(-1))
   #      else:
   #          x_flat = x.view(-1, x.size(-1))
   #      in_features = x_flat.size(-1)
   #
   #      if baseline is None:
   #          baseline = torch.zeros_like(x_flat)
   #
   #      sens_accum = torch.zeros(in_features, device=device)
   #      for _ in range(n_samples):
   #          noise = noise_sigma * torch.randn_like(x_flat)
   #          x_noisy = (x_flat + noise).detach()
   #
   #          alphas = torch.linspace(0.0, 1.0, steps, device=device)
   #          integrated_grad = torch.zeros_like(x_flat, device=device)
   #          for alpha in alphas:
   #              x_alpha = baseline + alpha * (x_noisy - baseline)
   #              x_alpha.requires_grad_(True)
   #              out = self.forward(x_alpha)
   #              out_sum = out.sum()
   #              grad = torch.autograd.grad(out_sum, x_alpha, retain_graph=False, create_graph=False)[0]
   #              grad = grad.clamp(-clip_val, clip_val)
   #              integrated_grad += grad
   #              x_alpha.requires_grad_(False)
   #
   #          avg_grad = integrated_grad / float(steps)
   #          ig = (x_noisy - baseline) * avg_grad
   #          sens_accum += ig.abs().mean(dim=0).detach()
   #
   #      sens = sens_accum / float(n_samples)
   #      sens = sens / (sens.max().clamp(min=eps))
   #      return sens.detach()
   #
   #  # 2️⃣ Jacobian 正则化损失
   #  def jacobian_regularization_loss(self, x_sample, lam=1e-3, mode='l1'):
   #      sens = self.GC_integrated_stable(x_sample)
   #      if mode == 'l1':
   #          reg = lam * sens.abs().mean()
   #      elif mode == 'l2':
   #          reg = lam * (sens ** 2).mean()
   #      else:
   #          raise ValueError("mode must be 'l1' or 'l2'")
   #      return reg
   #
   #  def GC_integrated_v11(self, x_sample, target_idx=None, baseline=None, steps=100, use_half=False, grad_clip=1.0):
   #      """
   #      Optimized Integrated Gradients for causal feature attribution.
   #      - Vectorized integration with Simpson weighting for accuracy.
   #      - Optional half precision and gradient clipping for efficiency/stability.
   #      """
   #      device = next(self.parameters()).device
   #      dtype = torch.float16 if use_half else torch.float32
   #
   #      # --- prepare x_flat ---
   #      x = x_sample.to(device, dtype=dtype)
   #      in_features = self.layers[0].in_features
   #      if x.size(-1) != in_features:
   #          raise ValueError(f"x_sample last dim must equal in_features={in_features}")
   #      x_flat = x.reshape(-1, in_features)
   #      N = x_flat.size(0)
   #
   #      # --- baseline setup ---
   #      if baseline is None:
   #          baseline = torch.zeros_like(x_flat, device=device, dtype=dtype)
   #      else:
   #          baseline = baseline.to(device, dtype=dtype).reshape(-1, in_features)
   #          if baseline.size(0) != N:
   #              baseline = baseline.expand_as(x_flat)
   #
   #      # --- prepare alpha and weights (Simpson integration) ---
   #      alphas = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
   #      weights = torch.ones_like(alphas)
   #      weights[1:-1:2] = 4.0  # Simpson's rule: 1,4,2,4,2,...,1
   #      weights[2:-1:2] = 2.0
   #      weights = weights / weights.sum()
   #
   #      # --- prepare integrated gradient accumulator ---
   #      total_grad = torch.zeros_like(x_flat, device=device, dtype=dtype)
   #
   #      # --- process all alpha points in mini-batches to save memory ---
   #      batch_size = min(steps, 32)
   #      for i in range(0, steps, batch_size):
   #          alpha_batch = alphas[i:i + batch_size].view(-1, 1, 1)
   #          x_alpha_batch = baseline.unsqueeze(0) + alpha_batch * (x_flat.unsqueeze(0) - baseline.unsqueeze(0))
   #          x_alpha_batch = x_alpha_batch.view(-1, in_features)
   #          x_alpha_batch.requires_grad_(True)
   #
   #          # forward + grad
   #          out = self.forward(x_alpha_batch)  # (batch*N, 1)
   #          out_sum = out.sum()
   #          grads = torch.autograd.grad(out_sum, x_alpha_batch, retain_graph=False, create_graph=False)[0]
   #          if grad_clip is not None:
   #              grads = grads.clamp_(-grad_clip, grad_clip)
   #
   #          grads = grads.view(len(alpha_batch), N, in_features)
   #          weighted_grad = (weights[i:i + batch_size].view(-1, 1, 1) * grads).sum(dim=0)
   #          total_grad += weighted_grad
   #
   #      # --- integrated gradients approximation ---
   #      ig = (x_flat - baseline) * total_grad
   #      causal_strength = ig.abs().mean(dim=0)
   #
   #      return causal_strength.detach().float()
   #  #去噪  重要方向 下面那个是去噪后的重要方向
   #  def GC_integrated_v2(self,
   #                       x_sample,
   #                       target_idx=None,
   #                       baseline=None,
   #                       steps=50,
   #                       denoise='smooth',  # 默认开启 SmoothGrad 去噪
   #                       n_samples=10,  # noise samples for smoothgrad (默认 10)
   #                       noise_sigma=None,  # scalar/array-like/torch tensor or None -> default 0.02*range
   #                       direction='grad',  # 默认使用梯度方向作为重要方向
   #                       reduce_method='abs_mean',  # 'abs_mean'|'mean'|'l2'
   #                       return_cpu=False  # if True return CPU tensor, else keep on model device
   #                       ):
   #      """
   #      Extended Integrated Gradients with default: SmoothGrad denoising + gradient-directional IG.
   #
   #      Returns:
   #          causal_strength: tensor shape (in_features,) (on model device by default, or CPU if return_cpu=True)
   #      """
   #      import torch
   #      device = next(self.parameters()).device
   #      x = x_sample.to(device)
   #
   #      # validate input last dim
   #      if x.dim() < 1 or x.size(-1) != self.layers[0].in_features:
   #          raise ValueError(f"x_sample last dim must equal in_features={self.layers[0].in_features}")
   #
   #      in_features = x.size(-1)
   #      x_flat = x.reshape(-1, in_features)  # (N, F)
   #      N = x_flat.size(0)
   #
   #      # baseline handling
   #      if baseline is None:
   #          baseline_flat = torch.zeros_like(x_flat, device=device)
   #      else:
   #          if not torch.is_tensor(baseline):
   #              baseline = torch.as_tensor(baseline)
   #          baseline_flat = baseline.to(device).reshape(-1, in_features)
   #          if baseline_flat.size(0) != x_flat.size(0):
   #              baseline_flat = baseline_flat.expand_as(x_flat)
   #
   #      # prepare noise_sigma when denoise == 'smooth'
   #      if denoise == 'smooth':
   #          if noise_sigma is None:
   #              xrng = (x_flat.max(dim=0).values - x_flat.min(dim=0).values).clamp(min=1e-6)
   #              noise_sigma = (0.02 * xrng).to(device)  # (F,)
   #          else:
   #              # normalize user-provided noise_sigma to tensor of shape (F,)
   #              if not torch.is_tensor(noise_sigma):
   #                  try:
   #                      noise_sigma = torch.as_tensor(noise_sigma, device=device)
   #                  except Exception:
   #                      noise_sigma = torch.tensor(float(noise_sigma), device=device)
   #              noise_sigma = noise_sigma.to(device)
   #              if noise_sigma.ndim == 0:
   #                  noise_sigma = noise_sigma.repeat(in_features)
   #              else:
   #                  noise_sigma = noise_sigma.reshape(-1)
   #                  if noise_sigma.numel() != in_features:
   #                      noise_sigma = noise_sigma.expand(in_features)
   #
   #      # helper: single IG run (xf, bf are (N,F)); dir_unit is None or (N,F) or (F,)
   #      def _integrated_gradients(xf, bf, dir_unit=None):
   #          integrated_grad = torch.zeros_like(xf, device=device)
   #          alphas = torch.linspace(0.0, 1.0, steps, device=device)
   #
   #          for alpha in alphas:
   #              if dir_unit is None:
   #                  x_alpha = bf + alpha * (xf - bf)  # straight-line IG (N,F)
   #              else:
   #                  diff = xf - bf  # (N,F)
   #                  if dir_unit.ndim == 1:
   #                      dir_u = dir_unit.unsqueeze(0).expand(N, -1)
   #                  else:
   #                      dir_u = dir_unit
   #                  p = (diff * dir_u).sum(dim=1, keepdim=True)  # (N,1)
   #                  x_alpha = bf + alpha * (dir_u * p)  # (N,F)
   #
   #              x_alpha.requires_grad_(True)
   #              out = self.forward(x_alpha)  # (N, out_features) expected out_features==1
   #              out_sum = out.sum()
   #              grads = torch.autograd.grad(out_sum, x_alpha, retain_graph=False, create_graph=False)[0]
   #              integrated_grad += grads
   #              x_alpha.requires_grad_(False)
   #
   #          avg_grad = integrated_grad / float(steps)
   #
   #          # multiply by same diff used for path construction
   #          if dir_unit is None:
   #              diff_for_mult = (xf - bf)
   #          else:
   #              diff = (xf - bf)
   #              if dir_unit.ndim == 1:
   #                  dir_u = dir_unit.unsqueeze(0).expand(N, -1)
   #              else:
   #                  dir_u = dir_unit
   #              p = (diff * dir_u).sum(dim=1, keepdim=True)  # (N,1)
   #              diff_for_mult = dir_u * p  # (N,F)
   #
   #          ig = diff_for_mult * avg_grad
   #          return ig.detach()
   #
   #      # compute direction unit(s) (默认 grad)
   #      dir_unit = None
   #      if direction is not None:
   #          if direction == 'grad':
   #              x_temp = x_flat.clone().detach().requires_grad_(True)
   #              out = self.forward(x_temp)
   #              out_sum = out.sum()
   #              grads_at_x = torch.autograd.grad(out_sum, x_temp, retain_graph=False, create_graph=False)[0]  # (N,F)
   #              norm = grads_at_x.norm(dim=1, keepdim=True).clamp(min=1e-8)
   #              dir_unit = grads_at_x / norm
   #          elif direction == 'pc1':
   #              diffs = (x_flat - baseline_flat).detach()
   #              if diffs.size(0) == 1:
   #                  v = diffs[0]
   #                  vn = v / (v.norm() + 1e-8)
   #                  dir_unit = vn.unsqueeze(0).expand(N, -1)
   #              else:
   #                  X = diffs - diffs.mean(dim=0, keepdim=True)
   #                  try:
   #                      U, S, Vt = torch.linalg.svd(X, full_matrices=False)
   #                      pc1 = Vt[0]
   #                      pc1 = pc1 / (pc1.norm() + 1e-8)
   #                      dir_unit = pc1.unsqueeze(0).expand(N, -1)
   #                  except Exception:
   #                      v = diffs.mean(dim=0)
   #                      dir_unit = (v / (v.norm() + 1e-8)).unsqueeze(0).expand(N, -1)
   #          else:
   #              raise ValueError("direction must be None|'grad'|'pc1'")
   #
   #      # main computation (默认使用 denoise='smooth')
   #      if denoise is None:
   #          ig = _integrated_gradients(x_flat, baseline_flat, dir_unit=dir_unit)
   #      elif denoise == 'smooth':
   #          total_ig = torch.zeros_like(x_flat, device=device)
   #          ns = noise_sigma.reshape(1, -1).expand(N, -1)
   #          for i in range(n_samples):
   #              noise = torch.randn_like(x_flat, device=device) * ns
   #              xf_noisy = (x_flat + noise).detach()
   #              # note: dir_unit derived from original x_flat; to recompute per-noise-sample you'd compute dir_unit inside loop
   #              total_ig += _integrated_gradients(xf_noisy, baseline_flat, dir_unit=dir_unit)
   #          ig = total_ig / float(n_samples)
   #      else:
   #          raise ValueError("denoise must be None or 'smooth'")
   #
   #      # aggregate across samples to per-feature causal strength
   #      if reduce_method == 'abs_mean':
   #          causal_strength = ig.abs().mean(dim=0)
   #      elif reduce_method == 'mean':
   #          causal_strength = ig.mean(dim=0)
   #      elif reduce_method == 'l2':
   #          causal_strength = (ig ** 2).mean(dim=0).sqrt()
   #      else:
   #          raise ValueError("reduce_method invalid")
   #
   #      if return_cpu:
   #          return causal_strength.detach().cpu()
   #      else:
   #          return causal_strength.detach()
   #
   #  def GC_integrated_modulated(self, x_sample, baseline=None, steps=50, beta=1.0):
   #      """
   #      Gradient-Modulated Granger Attribution (GMGA)
   #      - 结合结构因果权重 (GC) 与积分梯度敏感度。
   #      """
   #      device = next(self.parameters()).device
   #      x = x_sample.to(device)
   #      in_features = x.size(-1)
   #      x_flat = x.reshape(-1, in_features)
   #      N = x_flat.size(0)
   #
   #      # --- Step (a) GC structural prior ---
   #      GC_weight = self.GC()  # (in_features,)
   #      GC_prior = GC_weight.abs() / (GC_weight.abs().sum() + 1e-8)  # normalized prior
   #
   #      # --- Step (b) Path integral of gradients ---
   #      if baseline is None:
   #          baseline = torch.zeros_like(x_flat, device=device)
   #      else:
   #          baseline = baseline.to(device).reshape(-1, in_features)
   #
   #      integrated_grad = torch.zeros_like(x_flat, device=device)
   #      alphas = torch.linspace(0.0, 1.0, steps, device=device)
   #      for alpha in alphas:
   #          x_alpha = baseline + alpha * (x_flat - baseline)
   #          x_alpha.requires_grad_(True)
   #
   #          out = self.forward(x_alpha)
   #          out_sum = out.sum()
   #          grads = torch.autograd.grad(out_sum, x_alpha, retain_graph=False)[0]
   #          integrated_grad += grads
   #
   #      avg_grad = integrated_grad / float(steps)
   #      ig = (x_flat - baseline) * avg_grad
   #      grad_score = ig.abs().mean(dim=0)  # (in_features,)
   #
   #      # --- Step (c) Gradient-modulated fusion ---
   #      fused_score = torch.exp(beta * torch.log(GC_prior + 1e-8)) * grad_score
   #      fused_score = fused_score / (fused_score.sum() + 1e-8)  # normalized
   #
   #      return fused_score.detach()
   #
   #  def GC_integrated_v2(self, x_sample, target_idx=None, baseline=None, steps=50):
   #      """
   #      Integrated Gradients but keep the original input shape (e.g. (1, T, P)).
   #      Return per-source-feature attribution vector of length P (torch tensor on model device).
   #      """
   #      device = next(self.parameters()).device
   #      self_device = device
   #      x = x_sample.to(device)
   #
   #      # Expect x shape like (1, seq_len, P) as in training
   #      if x.dim() < 2:
   #          raise ValueError("x_sample must have at least 2 dims, expected (batch/1, seq_len, P)")
   #
   #      batch_dims = x.shape[:-1]  # all dims except last (e.g. (1, T))
   #      in_features = x.size(-1)  # P
   #
   #      # Prepare baseline with same shape
   #      x_flat = x.reshape(-1, in_features)  # (N, P) where N = product(batch_dims)
   #      N = x_flat.size(0)
   #      if baseline is None:
   #          baseline = torch.zeros_like(x, device=device)
   #      else:
   #          baseline = baseline.to(device)
   #          if baseline.shape != x.shape:
   #              # allow broadcasting from (1, P) -> (1, T, P) if user provided
   #              baseline = baseline.reshape(1, -1, in_features) if baseline.dim() == 2 else baseline
   #              if baseline.shape != x.shape:
   #                  raise ValueError("baseline shape incompatible with x_sample")
   #
   #      integrated_grad = torch.zeros_like(x, device=device)  # same shape as x
   #      alphas = torch.linspace(0.0, 1.0, steps, device=device)
   #
   #      # Ensure model doesn't run in torch.no_grad context outside
   #      # We'll explicitly require grad on x_alpha and use autograd.grad
   #      for alpha in alphas:
   #          x_alpha = baseline + alpha * (x - baseline)  # shape (1, T, P) or (batch, T, P)
   #          # make sure it's float and requires grad
   #          x_alpha = x_alpha.to(device).requires_grad_(True)
   #
   #          # forward - must be purely torch ops inside forward
   #          out = self.forward(x_alpha)  # should be tensor depending on x_alpha
   #          # out could be shape (batch, seq_len, 1) or (batch, seq_len) or (batch, 1)
   #          # sum all outputs to get scalar (aggregated over time/batch)
   #          out_sum = out.sum()
   #
   #          # Diagnostic: if this assertion fails, forward() disconnected grad
   #          if not out_sum.requires_grad:
   #              # gather small diagnostics to help you locate the problem
   #              msg = (
   #                  "GC_integrated_v2: out_sum does not require grad. "
   #                  "This means KAN.forward() detached the input or used non-torch ops.\n"
   #                  "Check KAN.forward(): remove .detach(), .data, .cpu(), .numpy() etc.\n"
   #              )
   #              # optional: print shapes to help debugging
   #              msg += f" x_alpha.requires_grad={x_alpha.requires_grad}, out.shape={out.shape}, device={device}\n"
   #              raise RuntimeError(msg)
   #
   #          # compute grads of scalar wrt x_alpha
   #          grads = torch.autograd.grad(out_sum, x_alpha, retain_graph=False, create_graph=False)[0]
   #          # grads shape same as x_alpha (batch, seq_len, P)
   #          integrated_grad += grads
   #
   #          # free memory: detach x_alpha (loop variable)
   #          x_alpha = x_alpha.detach()
   #
   #      avg_grad = integrated_grad / float(steps)  # (batch, seq_len, P)
   #      ig = (x - baseline) * avg_grad  # integrated gradients (batch, seq_len, P)
   #
   #      # Aggregate to per-feature attribution: mean over batch and time dims
   #      # Flatten leading dims except last
   #      ig_flat = ig.reshape(-1, in_features)  # (N, P)
   #      causal_strength = ig_flat.abs().mean(dim=0)  # (P,)
   #
   #      return causal_strength.detach()  # on device
   #
   #  # def update_GC_with_integrated(self, x_sample, beta=0.2, steps=50):
   #  #     """
   #  #     仅计算 Integrated Gradients 校正并保存到 self.GC_refined（不改写 base_weight）。
   #  #     返回：GC_refined (torch.tensor, device=cpu)
   #  #     """
   #  #     device = next(self.parameters()).device
   #  #     # 原始静态GC（P,)
   #  #     GC_static = self.GC().to(device)
   #  #
   #  #     # 计算 IG（P,)
   #  #     GC_IG = self.GC_integrated_v2(x_sample, steps=steps).to(device)
   #  #
   #  #     # 为避免尺度差异，我们把 GC_IG 缩放到和 GC_static 同的总能量（L2范数）
   #  #     eps = 1e-8
   #  #     scale = (GC_static.norm() + eps) / (GC_IG.norm() + eps)
   #  #     GC_IG_scaled = GC_IG * scale
   #  #
   #  #     # 融合（在原尺度下）
   #  #     GC_adjusted = (1.0 - beta) * GC_static + beta * GC_IG_scaled
   #  #
   #  #     # 只保存为解释缓存（CPU 上），不写回模型参数
   #  #     self.GC_refined = GC_adjusted.detach().cpu()
   #  #
   #  #     return self.GC_refined
   #  def update_gc_with_integrated(self, x_sample, baseline=None, steps=30, lambda_scale=0.05, writeback=True):
   #      """
   #      Compute IG-based causal strength vector and use it to adjust first-layer base_weight.
   #      If writeback=True, the adjusted weights are copied back into self.layers[0].base_weight
   #      Returns: the adjusted per-input vector (torch tensor on device)
   #      - lambda_scale: small multiplier controlling correction strength (recommended 0.01 - 0.2)
   #      """
   #      device = next(self.parameters()).device
   #      # ensure model in eval (no dropout) but gradients allowed for IG computation
   #      training_state = self.training
   #      self.eval()
   #
   #      # compute IG (ensures grads are computable)
   #      with torch.enable_grad():
   #          ig_vec = self.GC_integrated_v2(x_sample, baseline=baseline, steps=steps)  # (P,)
   #      # restore training state
   #      if training_state:
   #          self.train()
   #
   #      # stabilize scales: align IG to current base_weight norm
   #      with torch.no_grad():
   #          W = self.layers[0].base_weight  # shape likely (out_features, in_features) or (P, P)
   #          # we want a per-input vector with same device
   #          ig_vec = ig_vec.to(W.device)
   #
   #          # compute scale factors to map ig_vec to same L2 norm as per-input slices of W
   #          # compute target_norm = mean L2 of columns (over out_features) to stable per-input scale
   #          # handle if W is 1-D or 2-D
   #          if W.dim() == 1:
   #              target_norm = W.norm()
   #          else:
   #              # compute column norms (over output dim 0)
   #              col_norms = W.norm(p=2, dim=0)  # (in_features,)
   #              # to get single scale, use mean of column norms
   #              target_norm = col_norms.mean()
   #
   #          eps = 1e-8
   #          ig_norm = ig_vec.norm() + eps
   #          scale = (target_norm + eps) / ig_norm
   #
   #          ig_scaled = ig_vec * scale  # (P,)
   #
   #          # normalize ig_scaled to unitless correction factor
   #          # construct correction broadcastable to W shape
   #          correction = lambda_scale * (ig_scaled / (target_norm + eps))  # relative change factor (P,)
   #          # expand to W shape: broadcast on dim 0 (out_features)
   #          if W.dim() == 1:
   #              W_new = W * (1.0 + correction)
   #          else:
   #              W_new = W * (1.0 + correction.unsqueeze(0))
   #
   #          if writeback:
   #              # safe write-back to parameter (in-place)
   #              self.layers[0].base_weight.copy_(W_new)
   #
   #      # return the adjusted per-input vector (on cpu for logging convenience)
   #      return ig_scaled.detach()
   #
   #  def GC_attri(self, input_seq, target_idx):
   #      """
   #      使用梯度归因法计算从所有输入特征 -> target_idx 的因果强度
   #      input_seq: (1, T, P)
   #      target_idx: int, 表示预测哪个变量
   #      return: (P,) 因果强度
   #      """
   #      self.zero_grad()
   #      input_seq = input_seq.clone().detach().requires_grad_(True)
   #
   #      out = self(input_seq)  # (T,)
   #      out_target = out.mean()  # 或者只取最后时刻 out[-1]
   #      out_target.backward()
   #
   #      grads = input_seq.grad  # (1, T, P)
   #      avg_grads = grads.mean(dim=1).squeeze(0)  # (P,)
   #      avg_inputs = input_seq.mean(dim=1).squeeze(0)  # (P,)
   #
   #      # Gradient × Input
   #      importance = (avg_grads * avg_inputs).abs()
   #
   #      return importance
   #
   #
   #
   #
   #  def GC_grad_ge(self, input_seq, target_idx):
   #      """
   #      结合梯度敏感度和格兰杰因果思想的因果强度计算
   #      input_seq: (1, T, P)
   #      target_idx: int, 表示预测哪个变量
   #      return: (P,) 因果强度
   #      """
   #      self.zero_grad()
   #      input_seq = input_seq.clone().detach().requires_grad_(True)
   #
   #      # --- 前向传播 ---
   #      out = self(input_seq)  # (T, P)  假设输出是所有变量
   #      if out.ndim == 1:
   #          # 如果只输出 target 序列
   #          out_target = out[-1]
   #      else:
   #          out_target = out[-1, target_idx]  # 取最后时刻 target_idx 的预测值
   #
   #      # --- 反向传播 ---
   #      out_target.backward()
   #
   #      grads = input_seq.grad  # (1, T, P)
   #      avg_grads = grads.mean(dim=1).squeeze(0)  # (P,)
   #      avg_inputs = input_seq.mean(dim=1).squeeze(0)  # (P,)
   #
   #      # --- Gradient × Input (敏感度部分) ---
   #      importance_grad = (avg_grads * avg_inputs).abs()
   #
   #      # --- Granger-style 校正：考虑残差稳定性 ---
   #      # 对每个输入通道施加一个正则化系数，避免被无关噪声放大
   #      importance = importance_grad / (importance_grad.sum() + 1e-8)
   #
   #      return importance
   #
   #  import torch
   #
   #  def GC1(self, input_seq, target_idx, steps=20, use_mask=True):
   #      """
   #      改进版因果强度计算（适用于 KAN 或其他预测模型）
   #      - 支持 Integrated Gradients (更稳定)
   #      - 支持 时间加权 (避免平均抹平因果信号)
   #      - 支持 mask-out 残差对比 (更接近 Granger 因果)
   #
   #      Args:
   #          input_seq: (1, T, P) 输入序列
   #          target_idx: int，目标变量索引
   #          steps: int，IG 采样步数
   #          use_mask: bool，是否启用 mask-out 残差校正
   #
   #      Returns:
   #          importance: (P,) 每个输入变量对 target 的因果强度
   #      """
   #      self.zero_grad()
   #      input_seq = input_seq.clone().detach().requires_grad_(True)
   #      baseline = torch.zeros_like(input_seq)  # IG baseline
   #
   #      # --- Integrated Gradients ---
   #      grads_accum = 0
   #      for alpha in torch.linspace(0, 1, steps, device=input_seq.device):
   #          x = baseline + alpha * (input_seq - baseline)
   #          out = self(x)  # (T, P)
   #          out_target = out[-1, target_idx]
   #          grad = torch.autograd.grad(out_target, x, retain_graph=True)[0]  # (1, T, P)
   #          grads_accum += grad
   #
   #      avg_grads = grads_accum / steps  # (1, T, P)
   #
   #      # --- 时间加权聚合 ---
   #      T = input_seq.shape[1]
   #      time_weight = torch.linspace(1, T, T, device=input_seq.device) / T  # 越近权重越大
   #      weighted_grads = (avg_grads.squeeze(0) * time_weight[:, None]).sum(dim=0)  # (P,)
   #      avg_inputs = input_seq.mean(dim=1).squeeze(0)  # (P,)
   #
   #      # --- Gradient × Input (更稳定版) ---
   #      importance_grad = (weighted_grads * avg_inputs).abs()
   #
   #      # --- Mask-out 残差校正 (Granger 风味) ---
   #      if use_mask:
   #          with torch.no_grad():
   #              out_full = self(input_seq)[:, target_idx]  # (T,)
   #          residual_scores = []
   #          for j in range(input_seq.shape[2]):  # 遍历每个输入变量
   #              mask = torch.ones_like(input_seq)
   #              mask[:, :, j] = 0
   #              with torch.no_grad():
   #                  out_mask = self(input_seq * mask)[:, target_idx]
   #              residual_increase = (out_mask - out_full).pow(2).mean()
   #              residual_scores.append(residual_increase.item())
   #          residual_scores = torch.tensor(residual_scores, device=input_seq.device)
   #          importance_grad = importance_grad * (1 + residual_scores)
   #
   #      # --- 归一化 ---
   #      importance = importance_grad / (importance_grad.sum() + 1e-8)
   #
   #      return importance
   #
   #  # def GC(self, input_seq, M=4, probe='rademacher'):
   #  #     """
   #  #     Estimate per-input-feature causal strength using Hutchinson probes on the
   #  #     input->output Jacobian. Fast: uses M JvP/backward ops instead of P.
   #  #     Args:
   #  #         input_seq: Tensor with shape (batch, T, P) (your input_seq is (1,T,P))
   #  #         M: number of Hutchinson probes (4~8 is a good start)
   #  #         probe: 'rademacher' or 'normal'
   #  #     Returns:
   #  #         importance: Tensor shape (P,)  -- estimated L2 column-norm of Jacobian per input feature
   #  #     Notes:
   #  #         - This returns the sqrt of diag(J^T J) aggregated over time & batch,
   #  #           i.e. a sensible per-feature importance compatible with your pipeline.
   #  #     """
   #  #     device = input_seq.device
   #  #     # ensure leaf tensor that requires grad
   #  #     x = input_seq.clone().detach().requires_grad_(True)
   #  #
   #  #     # forward: model returns shape (batch, T, 1) for your KAN with out_features=1
   #  #     y = self(x)  # expected (batch, T, 1) or (batch, T) after squeeze
   #  #     y = y.squeeze(-1)  # -> (batch, T)
   #  #
   #  #     P = x.shape[-1]
   #  #     accum = x.new_zeros(P)
   #  #
   #  #     # use model.eval() style for stable behaviour (no dropout)
   #  #     was_training = self.training
   #  #     self.eval()
   #  #
   #  #     for m in range(M):
   #  #         if probe == 'rademacher':
   #  #             v = (torch.randint(0, 2, y.shape, device=device).float() * 2 - 1)
   #  #         else:
   #  #             v = torch.randn_like(y, device=device)
   #  #
   #  #         scalar = (v * y).sum()  # v^T y
   #  #         # grads = d (v^T y) / d x  -> shape same as x: (batch, T, P)
   #  #         grads = torch.autograd.grad(scalar, x, retain_graph=(m != M - 1))[0]
   #  #
   #  #         # accumulate squared grads across time and batch -> per-feature scalar
   #  #         # sum over time dimension, then mean over batch if batch>1
   #  #         # grads.pow(2).sum(dim=1) -> (batch, P)
   #  #         per_feature_sq = grads.pow(2).sum(dim=1).mean(dim=0)  # (P,)
   #  #         accum += per_feature_sq
   #  #
   #  #     # restore training mode
   #  #     if was_training:
   #  #         self.train()
   #  #
   #  #     accum = accum / float(M)  # unbiased estimate of diag(J^T J)
   #  #     importance = torch.sqrt(accum + 1e-16)  # return L2 norm per input feature
   #  #     return importance.detach()
   #
   #  # def GC(self, L):
   #  #     """
   #  #     输出每个源节点对目标节点的总因果强度 (长度 P)
   #  #     """
   #  #     GC = self.layers[0].base_weight  # shape: [out_features, P*L]
   #  #     hidden_size, PL = GC.shape
   #  #     P = PL // L
   #  #     W = GC.view(hidden_size, P, L)  # reshape: [hidden_size, P, L]
   #  #     weight_norm = torch.norm(W, dim=(0, 2))  # 对 hidden + L 求范数 -> [P,]
   #  #     return weight_norm
   #
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
