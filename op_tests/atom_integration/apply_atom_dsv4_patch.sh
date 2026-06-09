#!/usr/bin/env bash
# Apply the wo_a aiter patch to ATOM's deepseek_v4.py IN PLACE.
#
# Modifies /app/ATOM/atom/models/deepseek_v4.py to:
#  1. Skip the BF16 dequant in process_weights_after_loading; instead
#     convert weight_scale to UE8M0 u8 with aiter.convert_scales_to_ue8m0.
#  2. Replace the bf16 torch.einsum in forward with
#     aiter.batched_gemm_fp8_blockwise_einsum.
#
# Gated by env var so the original code path stays as a fallback:
#   AITER_DSV4_WO_A=1   -- enable the aiter path
#   unset / =0          -- keep the BF16 reference path
#
# Idempotent: re-running on an already-patched file is a no-op.
set -euo pipefail

TARGET="${1:-/app/ATOM/atom/models/deepseek_v4.py}"
PATCH_MARKER="# === AITER_DSV4_WO_A PATCH ==="

if grep -q "$PATCH_MARKER" "$TARGET"; then
    echo "Patch already applied to $TARGET — skipping."
    exit 0
fi

python <<'PYEOF'
import re, sys, os
target = os.environ.get("TARGET_FILE", "/app/ATOM/atom/models/deepseek_v4.py")
with open(target) as f:
    src = f.read()

MARKER = "# === AITER_DSV4_WO_A PATCH ==="

# 1. Replace the einsum block in forward.
#    Before:
#        o = o.view(num_tokens, self.n_local_groups, -1)
#        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
#        o = torch.einsum("sgd,grd->sgr", o, wo_a)
#        x = self.wo_b(o.flatten(1))
old_forward = '''        o = o.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        x = self.wo_b(o.flatten(1))'''
new_forward = '''        o = o.view(num_tokens, self.n_local_groups, -1)
        ''' + MARKER + ''' (forward) — aiter FP8 blockwise BMM
        if os.environ.get("AITER_DSV4_WO_A", "0") not in ("0", "", "false", "False") \\
                and getattr(self, "_aiter_wo_a_fp8", False):
            # FP8 path: A is BF16, quantize per-row to FP8 + scale, then aiter GEMM
            import aiter
            T, G, D = o.shape
            a_f32 = o.float().view(T, G, D // 128, 128)
            a_scale = (a_f32.abs().amax(dim=-1) / 448.0).clamp_min(1e-6)
            a_fp8 = (a_f32 / a_scale.unsqueeze(-1)).clamp_(-448.0, 448.0).view(T, G, D).to(torch.float8_e4m3fn)
            o_fp8 = aiter.batched_gemm_fp8_blockwise_einsum(
                "tgd,grd->tgr",
                a_fp8, a_scale,
                self.wo_a.weight, self.wo_a.weight_scale_inv,
                backend="auto",
            )
            x = self.wo_b(o_fp8.flatten(1))
        else:
            # Original BF16 reference path
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("sgd,grd->sgr", o, wo_a)
            x = self.wo_b(o.flatten(1))'''

if old_forward not in src:
    print("ERROR: forward block not found at expected position; aborting.")
    sys.exit(1)
src = src.replace(old_forward, new_forward, 1)

# 2. Wrap process_weights_after_loading: if AITER_DSV4_WO_A=1, skip dequant
#    and instead convert weight_scale -> uint8 in place.
old_pwal = '''    def process_weights_after_loading(self) -> None:'''
new_pwal = '''    def process_weights_after_loading(self) -> None:
        ''' + MARKER + ''' (load) — keep FP8 + convert W_scale -> u8
        import os as _os
        if _os.environ.get("AITER_DSV4_WO_A", "0") not in ("0", "", "false", "False"):
            import aiter
            if hasattr(self, "wo_a") and hasattr(self.wo_a, "weight_scale"):
                ws = self.wo_a.weight_scale
                if ws.dtype == torch.float32:
                    u8 = aiter.convert_scales_to_ue8m0(ws.data)
                    self.wo_a.weight_scale_inv = torch.nn.Parameter(u8, requires_grad=False)
                else:
                    self.wo_a.weight_scale_inv = self.wo_a.weight_scale
            # Mark wo_a as "use aiter FP8 path" for the forward.
            self._aiter_wo_a_fp8 = True
            self.wo_a.quant_type = QuantType.No  # prevent shuffle_weights
            return
        # Original BF16 dequant path:'''

if old_pwal not in src:
    print("ERROR: process_weights_after_loading signature not found; aborting.")
    sys.exit(1)
src = src.replace(old_pwal, new_pwal, 1)

# Ensure `import os` is present at module top (it usually is, but add to be safe).
if "\nimport os\n" not in src.split("class ")[0]:
    src = "import os\n" + src

with open(target, "w") as f:
    f.write(src)

print(f"Patched: {target}")
print(f"  Look for the marker '{MARKER}' in 2 places (forward + load).")
PYEOF
