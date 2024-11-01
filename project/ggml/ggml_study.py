import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import todos
import pdb
import ggml, ggml.utils
import ctypes
from typing import Optional

def ggml_tensor(ctx, t):
    '''t -- torch tensor'''
    n = t.cpu().to(torch.float).numpy()
    return ggml.utils.from_numpy(n, ctx)

def torch_tensor(g):
    '''g -- ggml tensor'''
    n = ggml.utils.to_numpy(g)
    return torch.from_numpy(n).clone()

def ggml_new_ctx(mem=256):
    params = ggml.ggml_init_params(mem_size=mem * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params)
    return ctx

def ggml_free_ctx(ctx):
    ggml.ggml_free(ctx)

def ggml_shape(prompt, g):
    print(f"{prompt} shape: ", g.contents.ne[:4]) # ggml.utils.get_shape(g))


def ggml_compute(ctx, f):
    gf = ggml.ggml_new_graph(ctx)
    ggml.ggml_build_forward_expand(gf, f)

    # Compute the graph
    ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)
# --------------------------------------------------------------------------------------

# ggml_tensor_t* ggml_nn_arange(ggml_context_t *ctx, ggml_tensor_t *x)
# {
#     int n = (int)ggml_nelements(x);

#     ggml_tensor_t *a = ggml_arange(ctx, 0.0, (float)n, 1.0);
#     a = ggml_scale(ctx, a, 1.0/(float)n);
#     x = ggml_cont(ctx, ggml_reshape_4d(ctx, a, x->ne[0], x->ne[1], x->ne[2], x->ne[3])); // !!! ggml_cont !!!

#     return x;
# }

def ggml_nn_arange(ctx, x):
    ne0 = x.contents.ne[0]
    ne1 = x.contents.ne[1]
    ne2 = x.contents.ne[2]
    ne3 = x.contents.ne[3]

    n = ggml.ggml_nelements(x)

    a = ggml.ggml_arange(ctx, 0.0, n, 1.0)
    a = ggml.ggml_scale(ctx, a, 1.0/n)
    x = ggml.ggml_cont(ctx, ggml.ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)) # !!! ggml_cont !!!

    # x = ggml.ggml_clamp(ctx, x, 1.0, 1.0)
    return x;

def torch_nn_arange3(x):
    B, C, H = x.size()
    a = torch.arange(x.nelement())/x.nelement()
    a = a.to(x.device)
    # a.fill_(1.0)

    return a.view(B, C, H)


def torch_nn_arange(x):
    B, C, H, W = x.size()
    a = torch.arange(x.nelement())/x.nelement()
    a = a.to(x.device)
    # a.fill_(1.0)

    return a.view(B, C, H, W)



def test_reshape(x):
    B, C, H, W = x.size()
    x = x.cpu().to(torch.float).numpy()

    # 1) Allocate a new context with 256 MB of memory
    params = ggml.ggml_init_params(mem_size=256 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params)

    input = ggml_tensor(ctx, x)
    # input = ggml.utils.from_numpy(x, ctx)
    print("ggml input shape: ", ggml.utils.get_shape(input))

    # 2) Build graph and compute
    f = ggml.ggml_reshape_4d(ctx, input, W//16, H//16, 256*C, B)
    gf = ggml.ggml_new_graph(ctx)
    ggml.ggml_build_forward_expand(gf, f)

    # 3) Compute the graph
    ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)
    # --------------------------------------------------------------------------------------

    # Get the output value
    print("ggml output shape: ", ggml.utils.get_shape(f))
    output = ggml.utils.to_numpy(f)
    output = torch.from_numpy(output).clone()

    # Free the context
    ggml.ggml_free(ctx)

    return output

def  ggml_nn_add(ctx, x, eps):
    g_eps = ggml.ggml_dup_tensor(ctx, x)
    g_eps = ggml.ggml_clamp(ctx, g_eps, eps, eps)
    return ggml.ggml_add(ctx, x, g_eps)


# @ggml.ggml_custom2_op_t
# def shuffle_map_function(
#     tensor_out: ggml.ggml_tensor_p,
#     a: ggml.ggml_tensor_p,
#     b: ggml.ggml_tensor_p,
#     ith: int,
#     nth: int,
#     userdata: Optional[ctypes.c_void_p],
# ):
#     R = 3
#     W, H, C, B = tensor_out.contents.ne[:4] # 12, 12, 1, 
#     dr = (B + nth - 1)//nth
#     start = ith * dr
#     stop = min(start + dr, B)

#     for batch in range(start, stop + 1):
#         for d_w  in range(W):
#             s_w = d_w // R
#             for d_h in range(H):
#                 s_h = d_h // R
#                 for d_c in range(C):
#                     s_c =  d_c * R * R + (d_h % R) * R + (d_w % R)
#                     value = ggml.ggml_get_f32_nd(b, s_w, s_h, s_c, batch)
#                     ggml.ggml_set_f32_nd(tensor_out, d_w, d_h, d_c, batch, value)


# # https://paperswithcode.com/method/pixelshuffle
# def ggml_pixel_shuffle(x, r=3):
#     B, C, H, W = x.size()

#     ctx = ggml_new_ctx()
#     a = ggml.ggml_new_tensor_4d(ctx, ggml.GGML_TYPE_F32, W * r, H * r, C//r//r, B)
#     b = ggml_tensor(ctx, x)

#     x_out = ggml.ggml_map_custom2(ctx, a, b, shuffle_map_function, ggml.GGML_N_TASKS_MAX, None)
#     ggml_compute(ctx, x_out)

#     output = torch_tensor(x_out)
#     ggml_free_ctx(ctx)

#     return output

# -----------------------------------------------------
def test_pixel_shuffle():
    # x = torch.randn(10, 9, 4, 4)
    x = torch.arange(10 * 9 * 4 * 4).reshape(10, 9, 4, 4)
    todos.debug.output_var("x", x)

    model = nn.PixelShuffle(3)
    with torch.no_grad():
        s = model(x)

    print("s torch.shape: ", s.shape)

    # y = ggml_pixel_shuffle(x, r=3)
    ctx = ggml_new_ctx()
    g_x = ggml_tensor(ctx, x)    

    g_y = ggml.ggml_shuffle(ctx, g_x, 3)
    ggml_compute(ctx, g_y)

    y = torch_tensor(g_y)

    ggml_free_ctx(ctx)

    todos.debug.output_var("y", y)
    todos.debug.output_var("(s - y).abs()", (s - y).abs())


# -----------------------------------------------------------------------------------
def ggml_pixel_unshuffle(x, r=16):
    B, C, H, W = x.size()

    # 1) Allocate a new context of memory
    ctx = ggml_new_ctx()

    g_x = ggml_tensor(ctx, x)
    ggml_shape("ggml g_x shape: ", g_x)

    # 2) Build graph
    a = ggml.ggml_new_tensor_4d(ctx, ggml.GGML_TYPE_F16,  r, r, C, C);
    g_y = ggml.ggml_im2col(ctx, a, g_x, r, r, 0, 0, 1, 1, True, ggml.GGML_TYPE_F32)
    ggml_shape("ggml g_y shape: ", g_y)
    # (192, 32, 32, 2) --> (32, 32, 192, 2)
    f = ggml.ggml_permute(ctx, g_y, 2, 0, 1, 3) # from src index to dst: 0->2, 1->0, 2->1, 3->3
    ggml_shape("ggml f shape: ", f)

    ggml_compute(ctx, f)
    output = torch_tensor(f)

    ggml_free_ctx(ctx)

    return output # output.size() -- [1, 64, 32, 32]

# ----------------------------------------------------------------------------------
def test_pixel_unshuffle():
    # x = torch.randn(5, 3, 256, 256)
    x = todos.data.load_tensor("images/out_sk.png")

    model = nn.PixelUnshuffle(16)
    with torch.no_grad():
        s = model(x)
    print("torch.shape: ", s.shape)
    y = ggml_pixel_unshuffle(x, r=16)
    print("ggml.shape: ", y.shape)
    todos.debug.output_var("y", y)
    
    todos.debug.output_var("(s - y).abs()", (s - y).abs())


# GGML_API struct ggml_tensor * ggml_mean(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);

def ggml_nn_mean(ctx, g_x, dim):
    dims = [0, 1, 2, 3]
    for i in range(dim + 1): # 0, 1, ..., dim_k
        if i < dim:
            dims[i] = i + 1
        else:
            dims[i] = 0 # i == dim_k
    g_x = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, g_x, dims[0], dims[1], dims[2], dims[3])) # [W, H, C, B] --> [C, W, H, B], from src to dst index: 0->1, 1->2, 2->0, 3->3

    g_mean0 = ggml.ggml_mean(ctx, g_x) # mean on dim 0

    for i in range(dim + 1): # 0, 1, ..., dim
        if i == 0:
            dims[i] = dim
        else:
            dims[i] = i - 1

    g_mean = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, g_mean0, dims[0], dims[1], dims[2], dims[3])) # [1, W, H, B] --> [W, H, 1, B], from src to dst index: 0->2, 1->0, 2->1, 3->3
    return g_mean    

# ------------------------------------------------------------------------------
def test_ggml_mean():
    x = torch.randn(2, 3, 256, 256)
    # B, C, H, W = x.size()
    dim_k = 2 # W, H, C, B, mean on channel ...
    mean_x = x.mean(dim = 3 - dim_k, keepdim=True)

    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    gx = ggml_tensor(ctx, x)
    gy = ggml_nn_mean(ctx, gx, dim_k)
    ggml_compute(ctx, gy)
    # --------------------------------------------------------------------------------------

    mean_y = torch_tensor(gy) # [2, 256, 256, 1]
    ggml_free_ctx(ctx)

    todos.debug.output_var("mean_x", mean_x)
    todos.debug.output_var("mean_y", mean_y)
    todos.debug.output_var("(mean.x - mean.y).abs()", (mean_x - mean_y).abs())

# --------------------------------------------------------------------------------------
def test_ggml_normalize():
    x = torch.randn(2, 3, 256, 256)
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    norm_x = (x - mean) / std

    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    g_x = ggml_tensor(ctx, x)
    g_mean = ggml.ggml_repeat(ctx, ggml_tensor(ctx, mean), g_x)
    g_std = ggml.ggml_repeat(ctx, ggml_tensor(ctx, std), g_x)
    # (g_x - g_mean)/g_std
    g_y = ggml.ggml_div(ctx, ggml.ggml_sub(ctx, g_x, g_mean), g_std)
    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    norm_y = torch_tensor(g_y)
    ggml_free_ctx(ctx)

    todos.debug.output_var("norm_x", norm_x)
    todos.debug.output_var("norm_y", norm_y)
    todos.debug.output_var("|norm_x - norm_y|", (norm_x - norm_y).abs())


# x1 = F.interpolate(x, size=(228, 228), mode="bilinear", align_corners=False)
def test_ggml_interpolate():
    x = torch.randn(2, 3, 1024, 1024)
    B, C, H, W = x.size()
    NH, NW = 511, 513
    upscale_x = F.interpolate(x, size=(NH, NW), mode="nearest")

    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    g_x = ggml_tensor(ctx, x)
    g_y = ggml.ggml_upscale_ext(ctx, g_x, NW, NH, C, B)
    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    upscale_y = torch_tensor(g_y)
    ggml_free_ctx(ctx)

    todos.debug.output_var("x", x)
    todos.debug.output_var("upscale_x", upscale_x)
    todos.debug.output_var("upscale_y", upscale_y)
    todos.debug.output_var("|upscale_x - upscale_y|", (upscale_x - upscale_y).abs())


def ggml_nn_slice(ctx, g_x, dim, start, stop, step):
    # n_dims = ggml.ggml_n_dims(g_x)
    # assert n_dims == 4, "x dimension must be 4"
    # # ------------------------

    starts = [0, 0, 0, 0]
    starts[dim] = max(start, 0)
    # print("starts: ", starts)
    # ------------------------
    stops = g_x.contents.ne[:4] # 4 -- n_dims
    stops[dim] = min(stop, stops[dim])
    # print("stops: ", stops)
    # ------------------------
    steps = [1, 1, 1, 1]
    steps[dim] = step
    # print("steps: ", steps)

    # ------------------------
    shapes = [0, 0, 0, 0]
    for i in range(4):
        shapes[i] = (stops[i] - starts[i] + steps[i] - 1) // steps[i]
    # print("shapes: ", shapes)

    # ------------------------
    strides = [0, 0, 0, 0]
    for i in range(4):
        strides[i] = g_x.contents.nb[i] * steps[i]
    # print("strides: ", strides)

    # ------------------------
    offset = 0
    for i in range(4):
        offset += g_x.contents.nb[i] * starts[i]

    # print("offset: ", offset)
    # ------------------------
    g_y = ggml.ggml_view_4d(ctx, g_x,
            shapes[0], shapes[1], shapes[2], shapes[3],
            strides[1], strides[2], strides[3],
            offset,
        )
    return g_y    

def ggml_nn_chunks(ctx, g_x, dim, k):
    B = g_x.contents.ne[:4][dim]
    S = (B + k - 1) // k
    chunks = []
    for i in range(k):
        # ggml_shape("g_x", g_x)
        # print(f"dim={dim}, B={B}, i = {i}, S={S}")
        c = ggml_nn_slice(ctx, g_x, dim, i * S, (i + 1)*S, 1)
        # ggml_shape("c", c)
        chunks.append(c)
    return chunks

# --------------------------------------------------------------------------------------
def test_ggml_slice():
    # x = torch.randn(2, 3, 256, 256)
    # slice_x = x[:, :, 1:256:2, :]
    x = torch.randn(256, 256)
    slice_x = x[1:256:2, :]

    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    dim = 1
    start = 1
    stop = 256
    step = 2

    g_x = ggml_tensor(ctx, x)
    g_y = ggml_nn_slice(ctx, g_x, dim, start, stop, step)
    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    slice_y = torch_tensor(g_y)

    c0, c1, c2, c3, c4 = ggml_nn_chunks(ctx, g_x, dim, 5)
    c0 = torch_tensor(c0)
    c1 = torch_tensor(c1)
    c2 = torch_tensor(c2)
    c3 = torch_tensor(c3)
    c4 = torch_tensor(c4)

    ggml_free_ctx(ctx)

    todos.debug.output_var("x", x)
    todos.debug.output_var("slice_x", slice_x)

    todos.debug.output_var("slice_y", slice_y)
    todos.debug.output_var("|slice_x - slice_y|", (slice_x - slice_y).abs())

    print("-" * 80)
    todos.debug.output_var("c0", c0)
    todos.debug.output_var("c1", c1)
    todos.debug.output_var("c2", c2)
    todos.debug.output_var("c3", c3)
    todos.debug.output_var("c4", c4)

# --------------------------------------------------------------------------------------
def test_ggml_batch_norm2d():
    num_features = 64
    x = torch.randn(2, num_features, 256, 256)
    B, C, H, W = x.size()

    # With Learnable Parameters ?
    bn = nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    state_dict = {}
    state_dict['weight'] = torch.randn(num_features)
    state_dict['bias'] = torch.randn(num_features)
    state_dict['running_mean'] = torch.randn(num_features)
    state_dict['running_var'] = torch.randn(num_features).clamp(0.0) # make sure >= 0.0
    bn.load_state_dict(state_dict)
    bn.eval()

    with torch.no_grad():
        batch_norm2d_x = bn(x)

    # batch_norm2d_y = (x - state_dict['running_mean'].reshape(1, C, 1, 1))/(state_dict['running_var'].reshape(1, C, 1, 1) + 1e-05).sqrt() * state_dict['weight'].reshape(1, C, 1, 1) + state_dict['bias'].reshape(1, C, 1, 1)

    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    g_x = ggml_tensor(ctx, x)
    g_weight = ggml_tensor(ctx, state_dict['weight'].reshape(1, C, 1, 1))
    g_bias = ggml_tensor(ctx, state_dict['bias'].reshape(1, C, 1, 1))

    g_running_mean = ggml_tensor(ctx, state_dict['running_mean'].reshape(1, C, 1, 1))
    # state_dict['running_var'] += 1e-5
    g_running_var = ggml_tensor(ctx, state_dict['running_var'].reshape(1, C, 1, 1))

    g_running_var = ggml_nn_add(ctx, g_running_var, 1e-5)
    g_running_var = ggml.ggml_sqrt(ctx, g_running_var)

    g_running_mean = ggml.ggml_repeat(ctx, g_running_mean, g_x)
    g_y = ggml.ggml_sub(ctx, g_x, g_running_mean)
    g_y = ggml.ggml_div(ctx, g_y, g_running_var)
    g_y = ggml.ggml_mul(ctx, g_y, g_weight)
    g_y = ggml.ggml_add(ctx, g_y, g_bias)

    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    batch_norm2d_y = torch_tensor(g_y)
    ggml_free_ctx(ctx)

    todos.debug.output_var("x", x)
    todos.debug.output_var("batch_norm2d_x", batch_norm2d_x)
    todos.debug.output_var("batch_norm2d_y", batch_norm2d_y)
    todos.debug.output_var("|batch_norm2d_x - batch_norm2d_y|", (batch_norm2d_x - batch_norm2d_y).abs())


# -------------------------------------------------------------------------------------
def test_ggml_batch_norm1d():
    num_features = 128
    x = torch.randn(2, num_features, 256)
    B, C, H = x.size()

    # With Learnable Parameters ?
    bn = nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    state_dict = {}
    # state_dict['weight'] = torch.randn(num_features).clamp(0.78, 1.36)
    # state_dict['bias'] = torch.randn(num_features).clamp(-0.13, 0.46)
    # state_dict['running_mean'] = torch.randn(num_features).clamp(-0.96, 19.16)
    state_dict['weight'] = torch.randn(num_features)
    state_dict['bias'] = torch.randn(num_features)
    state_dict['running_mean'] = torch.randn(num_features)
    state_dict['running_var'] = torch.randn(num_features).clamp(0.0)
    bn.load_state_dict(state_dict)
    bn.eval()

    with torch.no_grad():
        batch_norm1d_x = bn(x)

    # batch_norm2d_y = (x - state_dict['running_mean'].reshape(1, C, 1, 1))/(state_dict['running_var'].reshape(1, C, 1, 1) + 1e-05).sqrt() * state_dict['weight'].reshape(1, C, 1, 1) + state_dict['bias'].reshape(1, C, 1, 1)

    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    g_x = ggml_tensor(ctx, x)
    g_weight = ggml_tensor(ctx, state_dict['weight'].reshape(1, C, 1))
    g_bias = ggml_tensor(ctx, state_dict['bias'].reshape(1, C, 1))

    g_running_mean = ggml_tensor(ctx, state_dict['running_mean'].reshape(1, C, 1))
    # state_dict['running_var'] += 1e-5
    g_running_var = ggml_tensor(ctx, state_dict['running_var'].reshape(1, C, 1))
    g_running_var = ggml_nn_add(ctx, g_running_var, 1e-5)
    g_running_var = ggml.ggml_sqrt(ctx, g_running_var)

    g_running_mean = ggml.ggml_repeat(ctx, g_running_mean, g_x)
    g_y = ggml.ggml_sub(ctx, g_x, g_running_mean)
    g_y = ggml.ggml_div(ctx, g_y, g_running_var)
    g_y = ggml.ggml_mul(ctx, g_y, g_weight)
    g_y = ggml.ggml_add(ctx, g_y, g_bias)

    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    batch_norm1d_y = torch_tensor(g_y)
    ggml_free_ctx(ctx)

    todos.debug.output_var("x", x)
    todos.debug.output_var("batch_norm1d_x", batch_norm1d_x)
    todos.debug.output_var("batch_norm1d_y", batch_norm1d_y)
    todos.debug.output_var("|batch_norm1d_x - batch_norm1d_y|", (batch_norm1d_x - batch_norm1d_y).abs())


def ggml_nn_mul_mat(ctx, g_a, g_b):
    # a = torch.randn(2, 10, 3, 4)
    # b = torch.randn(2, 10, 4, 5)
    # g_a -- 4, 3, 10, 2
    # g_b -- 5, 4, 10, 2 --> 4, 5, 10, 2
    # ==> g_c -- 5, 3, 10, 2
    g_b = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, g_b, 1, 0, 2, 3))
    g_y = ggml.ggml_mul_mat(ctx, g_b, g_a)
    # ggml.ggml_mul_mat_set_prec(g_y, ggml.GGML_PREC_F32)

    return g_y

# https://ai.stackexchange.com/questions/40105/what-operation-is-ggml-mul-mat-performing-k%c3%97q-in-llama#:~:text=When%20you%20perform%20batched%20matrix%20multiplication,%20you%20multiply%202D%20matrices
# https://ai.stackexchange.com/questions/40105/what-operation-is-ggml-mul-mat-performing-k%c3%97q-in-llama
# --------------------------------------------------------------------------------------
def test_ggml_mul_mat():
    ''' batch matrix multi '''
    print("test_ggml_mul_mat ...")
    print(">" * 80)
    a = torch.randn(1, 8, 100, 1024)
    b = torch.randn(1, 8, 1024, 32)
    c_x = torch.matmul(a, b) # ==> [8, 100, 32]

    todos.debug.output_var("a", a)
    todos.debug.output_var("b", b)
    todos.debug.output_var("c_x", c_x)

    ctx = ggml_new_ctx()
    # --------------------------------------------------------------------------------------
    g_a = ggml_tensor(ctx, a) # ggml_shape("g_a", g_a)
    g_b = ggml_tensor(ctx, b) # ggml_shape("g_b", g_b)
    g_y = ggml_nn_mul_mat(ctx, g_a, g_b)

    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    c_y = torch_tensor(g_y)
    ggml_free_ctx(ctx)

    todos.debug.output_var("c_y", c_y) # [2, 10, 5, 3]
    todos.debug.output_var("|c_x - c_y|", (c_x - c_y).abs())
    print("<" * 80)

# --------------------------------------------------------------------------------------
def test_torch_bmm():
    ''' batch matrix multi '''
    print("test_torch_bmm ...")
    print(">" * 80)
    a = torch.randn(8, 100, 1024)
    b = torch.randn(8, 1024, 32)
    c_x = torch.bmm(a, b) # ==> [8, 100, 32]

    todos.debug.output_var("a", a)
    todos.debug.output_var("b", b)
    todos.debug.output_var("c_x", c_x)

    ctx = ggml_new_ctx()
    # --------------------------------------------------------------------------------------
    g_a = ggml_tensor(ctx, a) # ggml_shape("g_a", g_a)
    g_b = ggml_tensor(ctx, b) # ggml_shape("g_b", g_b)
    g_y = ggml_nn_mul_mat(ctx, g_a, g_b) # [32, 100, 8, 1]

    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    c_y = torch_tensor(g_y)
    ggml_free_ctx(ctx)

    todos.debug.output_var("c_y", c_y) # [2, 10, 5, 3]
    todos.debug.output_var("|c_x - c_y|", (c_x - c_y).abs())
    print("<" * 80)

# --------------------------------------------------------------------------------------
def test_ggml_einsum():
    print("test_ggml_einsum ...")
    print(">" * 80)
    a = torch.randn(1, 100, 256)
    b = torch.randn(1, 256, 512, 512)
    einsum_x = torch.einsum("bqc,bchw->bqhw", a, b) # [1, 100, 512, 512]
    todos.debug.output_var("a", a)
    todos.debug.output_var("b", b)
    todos.debug.output_var("einsum_x", einsum_x)

    m_a = a.reshape(100, 256)
    m_b = b.reshape(256, 512*512)

    ctx = ggml_new_ctx(mem=1024)
    # --------------------------------------------------------------------------------------
    g_a = ggml_tensor(ctx, m_a) # ggml_shape("g_a", g_a)
    g_b = ggml_tensor(ctx, m_b) # ggml_shape("g_b", g_b)

    g_c = ggml_nn_mul_mat(ctx, g_a, g_b)
    g_c = ggml.ggml_reshape_4d(ctx, g_c, 512, 512, 100, 1) # ggml_shape("g_c", g_c)

    ggml_compute(ctx, g_c)
    # --------------------------------------------------------------------------------------

    einsum_y = torch_tensor(g_c)
    ggml_free_ctx(ctx)

    todos.debug.output_var("einsum_y", einsum_y)
    todos.debug.output_var("|einsum_x - einsum_y|", (einsum_x - einsum_y).abs())
    print("<" * 80)



class PositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        # num_pos_feats = 128
        # temperature = 10000

        self.scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32) # [0.0, 127.0]
        self.dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats).reshape(1, 1, 1, 128)
        todos.debug.output_var("self.dim_t", self.dim_t)


    def forward(self, x):
        # x.size():  [1, 512, 32, 32] | [1, 512, 64, 64] | [1, 256, 128, 128]
        not_mask = torch.ones((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) # [1, 32, 32]
        grid_x = not_mask.cumsum(1, dtype=torch.float32)
        grid_y = not_mask.cumsum(2, dtype=torch.float32)

        # (Pdb) grid_x
        # tensor([[[ 1.,  1.,  1.,  ...,  1.,  1.,  1.],
        #          [ 2.,  2.,  2.,  ...,  2.,  2.,  2.],
        #          [ 3.,  3.,  3.,  ...,  3.,  3.,  3.],
        #          ...,
        #          [30., 30., 30.,  ..., 30., 30., 30.],
        #          [31., 31., 31.,  ..., 31., 31., 31.],
        #          [32., 32., 32.,  ..., 32., 32., 32.]]], device='cuda:0')
        # (Pdb) grid_x.size()
        # torch.Size([1, 32, 32])

        # normalize:
        eps = 1e-6
        grid_x = grid_x / (grid_x[:, -1:, :] + eps) * self.scale
        grid_y = grid_y / (grid_y[:, :, -1:] + eps) * self.scale
        todos.debug.output_var("grid_x 1==>", grid_x)
        todos.debug.output_var("grid_y 1==>", grid_y)

        pos_x = grid_x[:, :, :, None] / self.dim_t.to(x.device)
        pos_y = grid_y[:, :, :, None] / self.dim_t.to(x.device) # [1, 32, 32, 1] --> [1, 32, 32, 128]
        todos.debug.output_var("grid_x 2==>", pos_x)
        todos.debug.output_var("grid_y 2==>", pos_y)

        # y_sin = pos_x[:, :, :, 0::2].sin()
        # # tensor [y_sin] size: [2, 32, 32, 64], min: -1.0, max: 1.0, mean: 0.156305
        # y_cos = pos_x[:, :, :, 1::2].cos()
        # # tensor [y_cos] size: [2, 32, 32, 64], min: -1.0, max: 1.0, mean: 0.83215
        # # torch.stack((y_sin, y_cos), dim=4) # [2, 32, 32, 64, 2]

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # [2, 32, 32, 128]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        a_test = pos_y # [2, 32, 32, 128]

        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2) #  [2, 32, 32, 256] --> [2, 256, 32, 32]


        print("-" * 80)

        return pos, a_test


# @ggml.ggml_custom1_op_t
# def sin_cos_map(
#     tensor_out: ggml.ggml_tensor_p,
#     a: ggml.ggml_tensor_p,
#     ith: int,
#     nth: int,
#     userdata: Optional[ctypes.c_void_p],
# ):
#     N, W, H, B = tensor_out.contents.ne[:4] #     [128, 32, 32, 2]
#     dr = (B + nth - 1)//nth
#     start = ith * dr
#     stop = min(start + dr, B)

#     print(f"sin_cos_map: N = {N}, W = {W}, H = {H}, B = {B}")
#     ggml_shape("tensor_out", tensor_out)

#     for b_i in range(B): #range(start, stop):
#         for w_i in range(W):
#             for h_i in range(H):
#                 for n_i in range(N//2):
#                     sin_value1 =  ggml.ggml_get_f32_nd(a, 2*n_i + 0, w_i, h_i, b_i)
#                     cos_value1 =  ggml.ggml_get_f32_nd(a, 2*n_i + 1, w_i, h_i, b_i)
#                     sin_value2 = math.sin(sin_value1)
#                     cos_value2 = math.cos(cos_value1)
#                     # if w_i == W//2 or w_i == W//2 - 1 or w_i == W//2 + 1:
#                     #     print(f"===> sin_cos_map: w_i = {w_i}, sin_value1 = {sin_value1}, sin_value2 = {sin_value2}")
#                     ggml.ggml_set_f32_nd(tensor_out, 2*n_i + 0, w_i, h_i, b_i, sin_value2)
#                     ggml.ggml_set_f32_nd(tensor_out, 2*n_i + 1, w_i, h_i, b_i, cos_value2)

# --------------------------------------------------------------------------------------
def test_ggml_position_embedding():
    print("test_ggml_position_embedding ...")
    print(">" * 80)

    num_pos_feats = 128
    x = torch.randn(1, 256, 128, 128)
    B, C, H, W = x.size()
    model = PositionEmbedding(num_pos_feats) # 128
    model.eval()

    with torch.no_grad():
        pos_embed_x, a_test = model(x)
    todos.debug.output_var("x", x)
    todos.debug.output_var("pos_embed_x", pos_embed_x)

    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    print("*" * 80)
    g_x = ggml_tensor(ctx, x)
    g_y = ggml.ggml_arange(ctx, 1.0, (H + 1.0), 1.0)
    g_y = ggml.ggml_scale(ctx, g_y, 1.0/H * 2.0 * math.pi) # 32

    g_dim_t = ggml_tensor(ctx, model.dim_t) # [128, 1, 1, 1]

    a = ggml.ggml_new_tensor_4d(ctx, ggml.GGML_TYPE_F32, num_pos_feats, W, H, B)

    grid_x = ggml_nn_grid_x(ctx, g_y, W) # [32, 32, 1, 1]
    grid_x = ggml.ggml_reshape_4d(ctx, grid_x, 1, W, H, 1) # [1, 32, 32, 1]
    grid_x = ggml.ggml_repeat(ctx, grid_x, a)

    grid_y = ggml_nn_grid_y(ctx, g_y, H)
    grid_y = ggml.ggml_reshape_4d(ctx, grid_y, 1, W, H, 1)
    grid_y = ggml.ggml_repeat(ctx, grid_y, a)

    g_dim_t = ggml.ggml_repeat(ctx, g_dim_t, grid_x)
    pos_x = ggml.ggml_div(ctx, grid_x, g_dim_t)
    # pos_x = ggml.ggml_map_custom1(ctx, pos_x, sin_cos_map, ggml.GGML_N_TASKS_MAX, None)
    pos_x = ggml.ggml_sin_cos(ctx, pos_x)

    pos_y = ggml.ggml_div(ctx, grid_y, g_dim_t)
    # pos_y = ggml.ggml_map_custom1(ctx, pos_y, sin_cos_map, ggml.GGML_N_TASKS_MAX, None)
    pos_y = ggml.ggml_sin_cos(ctx, pos_y)

    g_pos_embed = ggml.ggml_concat(ctx, pos_x, pos_y, 0) 
    g_pos_embed = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, g_pos_embed, 2, 0, 1, 3))
    # ggml.shape: [256, 32, 32, 2] --> [32, 32, 256, 2]
    g_test = pos_y

    ggml_compute(ctx, g_pos_embed)
    # --------------------------------------------------------------------------------------

    pos_embed_y = torch_tensor(g_pos_embed)
    b_test = torch_tensor(g_test)

    ggml_free_ctx(ctx)
    todos.debug.output_var("pos_embed_y", pos_embed_y)

    print("*" * 80) # debug
    todos.debug.output_var("|pos_embed_x - pos_embed_y|", (pos_embed_x - pos_embed_y).abs())
    print("*" * 80) # debug
    todos.debug.output_var("a_test", a_test)
    todos.debug.output_var("b_test", b_test)
    todos.debug.output_var("|a_test - b_test|", (a_test.float() - b_test.float()).abs())
    print("<" * 80)


def ggml_nn_linear(ctx, g_x, g_w, g_b):
    # g_x = ggml.ggml_mul_mat(ctx, g_w, ggml.ggml_cont(ctx, g_x));
    # g_x = ggml.ggml_add(ctx, g_x, ggml.ggml_cont(ctx, g_b))

    g_x = ggml.ggml_mul_mat(ctx, g_w, g_x);
    g_x = ggml.ggml_add(ctx, g_x, g_b)

    return g_x

def in_projection_packed(q, k, v, w, b):
    # w.size() -- [1536, 512]
    # b.size() -- [1536]
    w_q, w_k, w_v = w.chunk(3)
    # (Pdb) w_q.size() -- [512, 512]
    # (Pdb) w_k.size() -- [512, 512]
    # (Pdb) w_v.size() -- [512, 512]

    b_q, b_k, b_v = b.chunk(3)
    # (Pdb) b_q.size(), b_k.size(), b_v.size() -- [512], [512], [512]
    k = k.to(q.dtype) # for half ?

    # torch.nn.functional.linear(input, weight, bias=None) → Tensor
    # y = F.linear(v, w_v, b_v)
    # tensor [v] size: [1024, 1, 256], min: -11.04214, max: 12.239643, mean: 0.041581
    # tensor [w_v] size: [256, 256], min: -0.072583, max: 0.07293, mean: 3.4e-05
    # tensor [b_v] size: [256], min: -0.00482, max: 0.003588, mean: -7e-05
    # tensor [y] size: [1024, 1, 256], min: -6.726958, max: 7.565886, mean: -0.01203
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v) # ggml_debug


def multi_head_attention_forward(query, key, value, num_heads,
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):

    # (Pdb) query.size() -- [100, 1, 256]
    # (Pdb) key.size() -- [1024, 1, 256]
    # (Pdb) value.size() -- [1024, 1, 256]

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape # torch.Size([100, 1, 256])
    head_dim = embed_dim // num_heads # 32 -- 256/8

    q, k, v = in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # tensor [q] size: [100, 1, 256], min: -3.425014, max: 3.135417, mean: -0.006095
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1) # [100, 8, 32] ==> [8, 100, 32]
    # tensor [q] size: [8, 100, 32], min: -3.425014, max: 3.135417, mean: -0.006095

    # tensor [k] size: [1024, 1, 256], min: -7.79951, max: 7.048189, mean: -0.051877
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    # tensor [k] size: [8, 1024, 32], min: -7.79951, max: 7.048189, mean: -0.051877

    # tensor [v] size: [1024, 1, 256], min: -6.726958, max: 7.565886, mean: -0.01203
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    # tensor [v] size: [8, 1024, 32], min: -6.726958, max: 7.565886, mean: -0.01203

    B, Nt, E = q.shape # [8, 100, 32]
    q_scaled = q / math.sqrt(E)

    # k.size() -- [8, 1024, 32]
    # k.transpose(-2, -1).size() -- [8, 32, 1024]
    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    # tensor [attn_output_weights] size: [8, 100, 1024], min: -6.5267, max: 8.720252, mean: 0.013913
    # todos.debug.output_var("attn_output_weights1", attn_output_weights)
    attn_output_weights = F.softmax(attn_output_weights, dim=-1) # [8, 100, 1024]
    # todos.debug.output_var("attn_output_weights2", attn_output_weights)
    # tensor [attn_output_weights1] size: [8, 100, 1024], min: -1614.528687, max: 1444.230591, mean: -0.565291
    # tensor [attn_output_weights2] size: [8, 100, 1024], min: 0.0, max: 1.0, mean: 0.000977

    # ==> attn_output_weights delta < 7.7e-05, v delta < 3.1e-05

    # tensor [attn_output_weights] size: [8, 100, 1024], min: -6.5267, max: 8.720252, mean: 0.013913
    # pp v.size() -- torch.Size([8, 1024, 32])
    # tgt_len * bsz, embed_dim -- 100, 256

    # xxxx_debug
    # attn_output_weights = torch_nn_arange3(attn_output_weights)
    # v = torch_nn_arange3(v)
    attn_output = torch.bmm(attn_output_weights, v)
    # tensor [attn_output] size: [8, 100, 32], min: -3.122495, max: 3.007122, mean: -0.015471
    # ===============================================================================
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim) # [8, 100, 32] ==> [100, 8, 32] ==> [100, 256]
    a_test = attn_output


    # (Pdb) out_proj_weight.size() -- [256, 256], out_proj_bias.size() -- [256]
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1)) # [100, 1, 256]
    # tensor [attn_output] size: [100, 1, 256], min: -2.726733, max: 2.047928, mean: -0.011761

    return attn_output, a_test # q_scaled

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim # 256
        self.num_heads = num_heads # 8
        self.in_proj_weight = torch.randn(3 * embed_dim, embed_dim) # nn.Parameter(torch.zeros((3 * embed_dim, embed_dim)))
        self.in_proj_bias = torch.randn(3 *embed_dim) # nn.Parameter(torch.zeros(3 * embed_dim))
        # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias) # bias == True
        self.out_proj_weight = torch.randn(embed_dim, embed_dim) # nn.Linear(embed_dim, embed_dim, bias=bias) # bias == True
        self.out_proj_bias = torch.randn(embed_dim) # nn.Linear(embed_dim, embed_dim, bias=bias) # bias == True

    def forward(self, query, key, value):
        # (Pdb) query.size() -- [100, 1, 256]
        # (Pdb) key.size() -- [1024, 1, 256]
        # (Pdb) value.size() -- [1024, 1, 256]
        attn_output, test = multi_head_attention_forward(
            query, key, value,
            self.num_heads,
            self.in_proj_weight, 
            self.in_proj_bias,
            # self.out_proj.weight, 
            # self.out_proj.bias,
            self.out_proj_weight, 
            self.out_proj_bias,
        )
        return attn_output, test # [100, 1, 256]

    def extra_repr(self) -> str:
        return f"embed_dim = {self.embed_dim}, num_heads = {self.num_heads}"

# ??? --------------------------------------------------------------------------------------
def test_ggml_multi_head_attention():
    print("test_ggml_multi_head_attention ...")
    print(">" * 80)
    query = torch.randn(100, 1, 256)
    key = torch.randn(1024, 1, 256)
    value = torch.randn(1024, 1, 256)

    model = MultiheadAttention(256, 8, True)
    model.eval()

    with torch.no_grad():
        attn_output_x, a_test = model(query, key, value)
    todos.debug.output_var("query", query)
    todos.debug.output_var("key", key)
    todos.debug.output_var("value", value)
    todos.debug.output_var("attn_output_x", attn_output_x)

    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    g_query = ggml_tensor(ctx, query)
    g_key = ggml_tensor(ctx, key)
    g_value = ggml_tensor(ctx, value)

    # --------------------------------------------------------------------------------------
    print("-" * 80)
    g_in_proj_weight = ggml_tensor(ctx, model.in_proj_weight) # [256, 768, 1, 1]
    g_in_proj_weight_q, g_in_proj_weight_k, g_in_proj_weight_v = ggml_nn_chunks(ctx, g_in_proj_weight, 1, 3)
    # g_in_proj_weight_q shape:  [256, 256, 1, 1]
    # g_in_proj_weight_k shape:  [256, 256, 1, 1]
    # g_in_proj_weight_v shape:  [256, 256, 1, 1]

    # --------------------------------------------------------------------------------------
    g_in_proj_bias = ggml_tensor(ctx, model.in_proj_bias) # [768, 1, 1, 1]
    g_in_proj_bias_q, g_in_proj_bias_k, g_in_proj_bias_v = ggml_nn_chunks(ctx, g_in_proj_bias, 0, 3)
    # g_in_proj_bias_q shape:  [256, 1, 1, 1]
    # g_in_proj_bias_k shape:  [256, 1, 1, 1]
    # g_in_proj_bias_v shape:  [256, 1, 1, 1]

    g_q = ggml_nn_linear(ctx, g_query, g_in_proj_weight_q, g_in_proj_bias_q)
    g_k = ggml_nn_linear(ctx, g_key, g_in_proj_weight_k, g_in_proj_bias_k)
    g_v = ggml_nn_linear(ctx, g_value, g_in_proj_weight_v, g_in_proj_bias_v)
    # g_q shape:  [256, 1, 100, 1]
    # g_k shape:  [256, 1, 1024, 1]
    # g_v shape:  [256, 1, 1024, 1]

    g_q = ggml.ggml_cont(ctx, ggml.ggml_reshape_3d(ctx, g_q, 32, 8, 100))
    g_q = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, g_q, 0, 2, 1, 3)) # [32, 100, 8, 1] 

    g_k = ggml.ggml_cont(ctx, ggml.ggml_reshape_3d(ctx, g_k, 32, 8, 1024))
    g_k = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, g_k, 0, 2, 1, 3)) # [32, 1024, 8, 1] 

    g_v = ggml.ggml_cont(ctx, ggml.ggml_reshape_3d(ctx, g_v, 32, 8, 1024))
    g_v = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, g_v, 0, 2, 1, 3)) # [32, 1024, 8, 1] 

    g_q_scaled = ggml.ggml_scale(ctx, g_q, 1.0/math.sqrt(32.0)) # [32, 1024, 8, 1] 
    # ==========================================================================================

    # # k.size() -- [8, 1024, 32]
    # # k.transpose(-2, -1).size() -- [8, 32, 1024]
    # attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    g_k = ggml.ggml_cont(ctx, ggml.ggml_transpose(ctx, g_k, 0, 1))
    attn_output_weights = ggml_nn_mul_mat(ctx, g_q_scaled, g_k)
    # [32, 100, 8, 1] x [1024, 32, 8, 1] --> [1024, 100, 8, 1]
    attn_output_weights = ggml.ggml_soft_max(ctx, attn_output_weights) # [1024, 100, 8, 1]
    # ================================================================================
    # g_q_scaled shape:  [32, 100, 8, 1]
    # g_k shape:  [1024, 32, 8, 1]
    # attn_output_weights shape:  [1024, 100, 8, 1]
    # attn_output_weights soft_max shape:  [1024, 100, 8, 1]
    # g_v shape: [32, 1024, 8, 1]
    # ==> [32, 100, 8, 1]
    # ================================================================================

    # attn_output = torch.bmm(attn_output_weights, v)
    # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim) # [8, 100, 32] ==> [100, 8, 32] ==> [100, 256]

    # attn_output_weights = ggml_nn_arange(ctx, attn_output_weights)
    # g_v = ggml_nn_arange(ctx, g_v)
    attn_output = ggml_nn_mul_mat(ctx, attn_output_weights, g_v) # [32, 100, 8, 1]
    attn_output = ggml.ggml_cont(ctx, ggml.ggml_permute(ctx, attn_output, 0, 2, 1, 3))
    attn_output = ggml.ggml_reshape_2d(ctx, attn_output, 256, 100) # [256, 100, 1, 1]
    g_test = attn_output 
    # xxxx_debug

    # --------------------------------------------------------------------------------
    g_out_proj_weight = ggml_tensor(ctx, model.out_proj_weight)
    g_out_proj_bias = ggml_tensor(ctx, model.out_proj_bias)
    # ********************************************************************************
    # attn_output shape:  [256, 100, 1, 1]
    # g_out_proj_weight shape:  [256, 256, 1, 1]
    # g_out_proj_bias shape:  [256, 1, 1, 1]
    # attn_output shape:  [256, 100, 1, 1]
    # ********************************************************************************
    # attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = ggml_nn_linear(ctx, attn_output, g_out_proj_weight, g_out_proj_bias)
    attn_output = ggml.ggml_cont(ctx, ggml.ggml_reshape_3d(ctx, attn_output, 256, 1, 100))
    # tensor [attn_output] size: [100, 1, 256], min: -2.726733, max: 2.047928, mean: -0.011761
    g_y = attn_output


    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    attn_output_y = torch_tensor(g_y)
    b_test = torch_tensor(g_test)

    ggml_free_ctx(ctx)
    todos.debug.output_var("attn_output_y", attn_output_y)
    todos.debug.output_var("(attn_output_x - attn_output_y).abs()", (attn_output_x - attn_output_y).abs())
    print("-" * 80)
    todos.debug.output_var("a_test", a_test)
    todos.debug.output_var("b_test", b_test)
    todos.debug.output_var("(a_test - b_test).abs()", (a_test - b_test).abs())
    print("<" * 80)

class LayerNormChannelsFirst(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        # self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.weight = torch.randn(normalized_shape)
        self.bias = torch.randn(normalized_shape)

        self.eps = eps
        self.normalized_shape = (normalized_shape, ) # 192 
    
    def forward(self, x):
        # tensor [x] size: [1, 192, 128, 128], min: -5.461643, max: 3.563172, mean: 0.154698
        u = x.mean(1, keepdim=True) # ggml_debug
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps) # ggml_debug

        # [192, 1, 1] * [1, 192, 128, 128] + [192, 1, 1]
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        # tensor [x] size: [1, 192, 128, 128], min: -6.796315, max: 7.445028, mean: -0.00847

        a_test = x

        return x, a_test

    def extra_repr(self) -> str:
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}'

# ----------------------------------------------------------------------
def test_ggml_layer_norm_channels_first():
    print("")
    print(">" * 80)
    x = torch.randn(1, 192, 128, 128)
    B, C, H, W = x.size()

    model = LayerNormChannelsFirst(192) 
    model.eval()

    with torch.no_grad():
        first_x, a_test = model(x)
    todos.debug.output_var("first_x", first_x)


    ctx = ggml_new_ctx(mem=1024)


    # --------------------------------------------------------------------------------------
    g_x = ggml_tensor(ctx, x)
    g_u = ggml_nn_mean(ctx, g_x, 2) # dim == 2
    g_u = ggml.ggml_repeat(ctx, g_u, g_x)
    g_d = ggml.ggml_sub(ctx, g_x, g_u)

    # g_var + eps ...
    g_s = ggml.ggml_mul(ctx, g_d, g_d)
    g_s = ggml_nn_mean(ctx, g_s, 2) # dim = 2
    g_s = ggml_nn_add(ctx, g_s, 1e-6)
    g_s = ggml.ggml_sqrt(ctx, g_s)
    g_s = ggml.ggml_repeat(ctx, g_s, g_d)

    g_x = ggml.ggml_div(ctx, g_d, g_s)


    # g_weight = ggml_tensor(ctx, model.weight.reshape(192, 1, 1).repeat(1, 128, 128))
    g_weight = ggml_tensor(ctx, model.weight) # [192, 1, 1, 1]
    g_weight = ggml.ggml_reshape_4d(ctx, g_weight, 1, 1, 192, 1) #  # [1, 1, 192, 1]
    g_weight = ggml.ggml_repeat(ctx, g_weight, g_x)
    # ggml_shape("g_weight3 ===== ", g_weight) 


    # g_bias = ggml_tensor(ctx, model.bias.reshape(192, 1, 1).repeat(1, 128, 128))
    g_bias = ggml_tensor(ctx, model.bias)
    g_bias = ggml.ggml_reshape_4d(ctx, g_bias, 1, 1, 192, 1)
    g_bias = ggml.ggml_repeat(ctx, g_bias, g_x)

    g_y = ggml.ggml_mul(ctx, g_x, g_weight)
    g_y = ggml.ggml_add(ctx, g_y, g_bias)

    g_test = g_y

    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    first_y = torch_tensor(g_y) # [2, 256, 256, 1]
    b_test = torch_tensor(g_test)
    ggml_free_ctx(ctx)

    print("-" * 80)
    todos.debug.output_var("a_test", a_test)
    todos.debug.output_var("b_test", b_test)
    todos.debug.output_var("|a_test - b_test|", (a_test - b_test).abs())
    print("-" * 80)

    todos.debug.output_var("first_y", first_y)
    todos.debug.output_var("|first_x - first_y|", (first_x - first_y).abs())
    print(">" * 80)


# -----------------------------------------------
def test_ggml_add(eps):
    x = torch.randn(2, 3, 256, 256)
    # eps = 1e-1
    torch_x = x + eps

    ctx = ggml_new_ctx(mem=1024)

    g_x = ggml_tensor(ctx, x)
    # -----------------------------------------------
    g_y = ggml.ggml_dup_tensor(ctx, g_x)
    ggml.ggml_set_f32(g_y, eps)
    g_y = ggml.ggml_add(ctx, g_x, g_y)
    # -----------------------------------------------
    ggml_compute(ctx, g_y)

    ggml_x = torch_tensor(g_y) 
    ggml_free_ctx(ctx)


    todos.debug.output_var("torch_x", torch_x)
    todos.debug.output_var("ggm_x", ggml_x)
    print("|torch_x - ggml_x|", (torch_x - ggml_x).abs())

# -----------------------------------------------------------------------------------
def test_ggml_nn_softmax():
    print("test_ggml_nn_softmax ...")
    print(">" * 80)
    x = torch.randn(2, 3, 256, 256)
    # x = torch.randn(8, 32, 1024)
    torch_x = F.softmax(x, dim = -1)

    ctx = ggml_new_ctx(mem=1024)

    g_x = ggml_tensor(ctx, x)
    # -----------------------------------------------
    g_y = ggml.ggml_soft_max(ctx, g_x)
    # -----------------------------------------------
    ggml_compute(ctx, g_y)

    ggml_x = torch_tensor(g_y) 
    ggml_free_ctx(ctx)

    todos.debug.output_var("torch_x", torch_x)
    todos.debug.output_var("ggm_x", ggml_x)
    todos.debug.output_var("(torch_x - ggml_x).abs()", (torch_x - ggml_x).abs())
    print("<" * 80)


# -----------------------------------------------
def test_ggml_repeat():
    print("test_ggml_repeat ...")
    print(">" * 80)
    x = torch.randn(2, 3, 256, 256)
    torch_x = x.repeat(2, 3, 4, 5)

    ctx = ggml_new_ctx(mem=1024)

    g_x = ggml_tensor(ctx, x)
    # -----------------------------------------------
    W, H, C, B = g_x.contents.ne[:4]
    g_t = ggml.ggml_new_tensor_4d(ctx, ggml.GGML_TYPE_F32, W * 5, H * 4, C * 3, B * 2)
    g_y = ggml.ggml_repeat(ctx, g_x, g_t)
    # -----------------------------------------------
    ggml_compute(ctx, g_y)

    ggml_x = torch_tensor(g_y) 
    ggml_free_ctx(ctx)

    todos.debug.output_var("torch_x", torch_x)
    todos.debug.output_var("ggm_x", ggml_x)
    todos.debug.output_var("|torch_x - ggml_x|", (torch_x - ggml_x).abs())
    print("<" * 80)



def ggml_nn_avgpool2d(ctx, x, kernel_size, stride_size):
    pad_size = kernel_size - stride_size
    W = x.contents.ne[0]
    H = x.contents.ne[1]
    C = x.contents.ne[2]
    B = x.contents.ne[3]

    # x = ggml.ggml_cont(ctx, ggml.ggml_pad(ctx, x, pad_size, pad_size, 0, 0)) # W+1, H+1, C, B
    # x = ggml.ggml_cont(ctx, x)

    x = ggml.ggml_cont(ctx, ggml.ggml_reshape_3d(ctx, x, W, H, C*B))
    y = ggml.ggml_cont(ctx, ggml.ggml_pool_2d(ctx, x, ggml.GGML_OP_POOL_AVG, kernel_size, kernel_size, stride_size, stride_size, pad_size, pad_size))
    y = ggml.ggml_cont(ctx, ggml.ggml_reshape_4d(ctx, y, W + pad_size, H + pad_size, C, B))
    y = ggml_nn_slice(ctx, y, 0, 1, W - pad_size, 1)
    y = ggml_nn_slice(ctx, y, 1, 1, H - pad_size, 1)

    return y;


# -----------------------------------------------
def test_ggml_pool_2d():
    print("test_ggml_pool_2d ...")
    print(">" * 80)
    kernel_size = 2
    stride_size = 1
    x = torch.randn(10, 3, 256, 256)
    B, C, H, W = x.size()

    model = nn.AvgPool2d(kernel_size, stride=stride_size)
    model.eval()

    with torch.no_grad():
        torch_x = model(x)

    todos.debug.output_var("torch_x", torch_x)

    ctx = ggml_new_ctx(mem=1024)
    g_x = ggml_tensor(ctx, x)
    # -----------------------------------------------
    # // GGML_API struct ggml_tensor * ggml_pool_2d(
    # //         struct ggml_context * ctx,
    # //         struct ggml_tensor  * a,
    # //         enum ggml_op_pool     op,
    # //         int                   k0,
    # //         int                   k1,
    # //         int                   s0,
    # //         int                   s1,
    # //         float                 p0,
    # //         float                 p1);
    g_x = ggml.ggml_reshape_3d(ctx, g_x, W, H, C*B)
    g_y = ggml.ggml_pool_2d(ctx, g_x, ggml.GGML_OP_POOL_AVG, kernel_size, kernel_size, stride_size, stride_size, 1, 1)
    g_y = ggml.ggml_reshape_4d(ctx, g_y, W + 1, H + 1, C, B)

    g_y = ggml_nn_slice(ctx, g_y, 0, 1, W, 1)
    g_y = ggml_nn_slice(ctx, g_y, 1, 1, H, 1)

    # g_y = ggml_nn_avgpool2d(ctx, g_x, kernel_size, stride_size)

    # -----------------------------------------------
    ggml_compute(ctx, g_y)

    ggml_x = torch_tensor(g_y) 
    ggml_free_ctx(ctx)

    todos.debug.output_var("ggm_x", ggml_x)
    todos.debug.output_var("|torch_x - ggml_x|", (torch_x - ggml_x).abs())
    print("<" * 80)


def ggml_nn_grid_y(ctx, x, H):
    W = x.contents.ne[0]
    out_shape = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, W, H) # H -- Repeat rows ...
    grid_y = ggml.ggml_repeat(ctx, x, out_shape)
    return ggml.ggml_cont(ctx, grid_y)


def ggml_nn_grid_x(ctx, x, n):
    grid_y = ggml_nn_grid_y(ctx, x, n)
    grid_x = ggml.ggml_transpose(ctx, grid_y)
    return ggml.ggml_cont(ctx, grid_x)

# -----------------------------------------------
def test_ggml_grid_x():
    print("test_ggml_grid_x ...")
    print(">" * 80)
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6, 7, 8])
    torch_grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    todos.debug.output_var("torch_grid_x", torch_grid_x)
    print(torch_grid_x)

    # -----------------------------------------------
    ctx = ggml_new_ctx(mem=1024)
    x = ggml.ggml_arange(ctx, 1.0, 4.0, 1.0) # [32, 1, 1, 1]
    grid_x = ggml_nn_grid_x(ctx, x, y.size(0))
    # -----------------------------------------------
    ggml_compute(ctx, grid_x)

    ggml_grid_x = torch_tensor(grid_x) 
    ggml_free_ctx(ctx)

    todos.debug.output_var("ggml_grid_x", ggml_grid_x)
    todos.debug.output_var("|torch_grid_x - ggml_grid_x|", (torch_grid_x - ggml_grid_x).abs())

    # print(ggml_grid_x)
    print(">" * 80)

# -----------------------------------------------
def test_depthwise_conv():
    print("test_depthwise_conv ...")
    print(">" * 80)
    dim = 192
    dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
    # dwconv.weight -- [192, 1, 7, 7]
    # dwconv.bias -- [192]
    state_dict = {}
    state_dict['weight'] = torch.randn(dim, 1, 7, 7)
    state_dict['bias'] = torch.randn(dim)
    dwconv.load_state_dict(state_dict)
    dwconv.eval()

    x = torch.randn(2, dim, 256, 256)
    with torch.no_grad():
        torch_x = dwconv(x)

    ctx = ggml_new_ctx(mem=4096)

    g_x = ggml_tensor(ctx, x)
    # -----------------------------------------------
    g_w = ggml_tensor(ctx, state_dict['weight'])
    g_w = ggml.ggml_cast(ctx, g_w, ggml.GGML_TYPE_F16)
    g_b = ggml_tensor(ctx, state_dict['bias'])

    # g_y = ggml.ggml_conv_depthwise_2d(ctx, g_w, g_x, s0, s1, p0, p1, d0, d1)
    g_y = ggml.ggml_conv_depthwise_2d(ctx, g_w, g_x, 1, 1, 3, 3, 1, 1)
    g_b = ggml.ggml_reshape_4d(ctx, g_b, 1, 1, dim, 1);
    g_y = ggml.ggml_add(ctx, g_y, g_b);

    # -----------------------------------------------
    ggml_compute(ctx, g_y)

    ggml_x = torch_tensor(g_y) 
    ggml_free_ctx(ctx)

    todos.debug.output_var("torch_x", torch_x)
    todos.debug.output_var("ggm_x", ggml_x)
    todos.debug.output_var("|torch_x - ggml_x|", (torch_x - ggml_x).abs())
    print("<" * 80)

# -----------------------------------------------
def test_ggml_mul_mat2():
    '''matrix dot multi'''
    print("test_ggml_mul_mat2 ...")
    print(">" * 80)

    dim = 192
    gamma = torch.randn(dim)
    x = torch.randn(2, 128, 128, dim) # [1, 128, 128, 192]
    c_x = gamma * x

    todos.debug.output_var("gamma", gamma)
    todos.debug.output_var("x", x)
    todos.debug.output_var("c_x", c_x)


    ctx = ggml_new_ctx()
    # --------------------------------------------------------------------------------------
    g_gamma = ggml_tensor(ctx, gamma) # ggml_shape("g_a", g_a)
    g_x = ggml_tensor(ctx, x) # ggml_shape("g_a", g_a)

    g_gamma = ggml.ggml_repeat(ctx, g_gamma, g_x)
    g_y = ggml.ggml_mul(ctx, g_gamma, g_x)
    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    c_y = torch_tensor(g_y)
    ggml_free_ctx(ctx)

    todos.debug.output_var("c_y", c_y) # [2, 10, 5, 3]
    todos.debug.output_var("|c_x - c_y|", (c_x - c_y).abs())
    print("<" * 80)

# -----------------------------------------------
def test_ggml_nn_linear():
    print("test_ggml_nn_linear ...")
    print(">" * 80)

    dim = 192
    model = nn.Linear(dim, 4*dim)
    state_dict = {}
    state_dict['weight'] = torch.randn(4*dim, dim)
    state_dict['bias'] = torch.randn(4*dim)
    model.load_state_dict(state_dict)
    model.eval()

    x = torch.randn(2, 128, 128, dim)
    with torch.no_grad():
        torch_x = model(x)

    ctx = ggml_new_ctx(mem=4096)

    g_x = ggml_tensor(ctx, x)
    # -----------------------------------------------
    g_w = ggml_tensor(ctx, state_dict['weight'])
    g_b = ggml_tensor(ctx, state_dict['bias'])

    # g_y = ggml.ggml_mul_mat(ctx, g_w, ggml.ggml_cont(ctx, g_x))
    g_y = ggml.ggml_mul_mat(ctx, g_w, g_x)
    g_y = ggml.ggml_add(ctx, g_y, g_b);

    # -----------------------------------------------
    ggml_compute(ctx, g_y)

    ggml_x = torch_tensor(g_y) 
    ggml_free_ctx(ctx)

    todos.debug.output_var("torch_x", torch_x)
    todos.debug.output_var("ggm_x", ggml_x)
    todos.debug.output_var("|torch_x - ggml_x|", (torch_x - ggml_x).abs())
    print("<" * 80)

class LayerNormChannelsLast(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        # self.normalized_shape=(192,), self.eps=1e-06
        # tensor [x] size: [1, 128, 128, 192], min: -2.912116, max: 3.049789, mean: 0.000583
        # tensor [self.weight] size: [192], min: -0.653009, max: 4.843045, mean: 0.762637
        # tensor [self.bias] size: [192], min: -2.733021, max: 1.236647, mean: 0.005043
        # y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # tensor [y] size: [1, 128, 128, 192], min: -18.163292, max: 15.625493, mean: 0.006303

        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps) # ggml_debug

    def extra_repr(self) -> str:
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}'

# --------------------------------------------------------------------------------------
def test_ggml_layer_norm_channels_last():
    print("test_ggml_layer_norm_channels_last ")
    print(">" * 80)
    num_features = 64

    x = torch.randn(1, 128, 128, num_features)
    # x = torch_nn_arange(x)
    
    model = LayerNormChannelsLast(num_features)
    state_dict = {}
    state_dict['weight'] = torch.randn(num_features)
    state_dict['bias'] = torch.randn(num_features)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        last_x = model(x)
    todos.debug.output_var("last_x", last_x)

    ctx = ggml_new_ctx(mem=4096)

    g_x = ggml_tensor(ctx, x)
    g_weight = ggml_tensor(ctx, state_dict['weight'])
    g_bias = ggml_tensor(ctx, state_dict['bias'])


    # ggml_tensor_t* ggml_nn_layer_norm(ggml_context_t* ctx, ggml_tensor_t* x, ggml_tensor_t* w, ggml_tensor_t* b, float eps)
    # {
    #     x = ggml_norm(ctx, x, eps);
    #     x = ggml_mul(ctx, x, w);
    #     x = ggml_add(ctx, x, b);
    #     return x;
    # }
    # --------------------------------------------------------------------------------------
    # g_y = ggml.ggml_norm(ctx, g_x, model.eps)
    g_u = ggml.ggml_mean(ctx, g_x) # ggml_nn_mean(ctx, g_x, 0) # ggml.ggml_mean(ctx, g_x)
    # g_u = ggml.ggml_repeat(ctx, g_u, g_x)
    g_d = ggml.ggml_sub(ctx, g_x, g_u)

    g_s = ggml.ggml_mul(ctx, g_d, g_d)
    g_s = ggml.ggml_mean(ctx, g_s) # ggml_nn_mean(ctx, g_s, 0) # ggml.ggml_mean(ctx, g_s)
    g_s = ggml_nn_add(ctx, g_s, model.eps)
    g_s = ggml.ggml_sqrt(ctx, g_s)

    g_x = ggml.ggml_div(ctx, g_d, g_s)

    # g_x shape:  [192, 128, 128, 2]
    # g_y shape:  [192, 128, 128, 2]
    g_y = ggml.ggml_mul(ctx, g_x, g_weight)
    g_y = ggml.ggml_add(ctx, g_y, g_bias)
    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    last_y = torch_tensor(g_y) # [2, 256, 256, 1]
    ggml_free_ctx(ctx)

    print("-" * 80)
    todos.debug.output_var("last_y", last_y)
    todos.debug.output_var("|last_x - last_y|", (last_x - last_y).abs())
    print("<" * 80)

# --------------------------------------------------------------------------------------
def test_ggml_layer_norm():
    print("test_ggml_layer_norm ...")
    print(">" * 80)

    x = torch.randn([100, 1, 256])

    model = nn.LayerNorm(256);
    state_dict = {}
    state_dict['weight'] = torch.randn(256)
    state_dict['bias'] = torch.randn(256)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        last_x = model(x)
    todos.debug.output_var("last_x", last_x)

    ctx = ggml_new_ctx(mem=1024)

    g_x = ggml_tensor(ctx, x)
    g_weight = ggml_tensor(ctx, state_dict['weight'])
    g_bias = ggml_tensor(ctx, state_dict['bias'])

    # --------------------------------------------------------------------------------------
    # g_y = ggml.ggml_norm(ctx, g_x, model.eps)
    g_u = ggml.ggml_mean(ctx, g_x)
    g_u = ggml.ggml_repeat(ctx, g_u, g_x)
    g_d = ggml.ggml_sub(ctx, g_x, g_u)

    g_s = ggml.ggml_mul(ctx, g_d, g_d)
    g_s = ggml.ggml_mean(ctx, g_s)
    g_s = ggml_nn_add(ctx, g_s, model.eps)
    g_s = ggml.ggml_sqrt(ctx, g_s)

    g_x = ggml.ggml_div(ctx, g_d, g_s)

    # g_x shape:  [192, 128, 128, 2]
    # g_y shape:  [192, 128, 128, 2]
    g_y = ggml.ggml_mul(ctx, g_x, g_weight)
    g_y = ggml.ggml_add(ctx, g_y, g_bias)
    ggml_compute(ctx, g_y)
    # --------------------------------------------------------------------------------------

    last_y = torch_tensor(g_y) # [2, 256, 256, 1]
    ggml_free_ctx(ctx)

    print("-" * 80)
    todos.debug.output_var("last_y", last_y)

    todos.debug.output_var("|last_x - last_y|", (last_x - last_y).abs())
    print("<" * 80)


class CustomAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)

    def forward(self, x):
        # todos.debug.output_var("x1", x)
        x = self.pad(x)
        # todos.debug.output_var("x2", x)

        x = self.blur(x)
        # todos.debug.output_var("x3", x)

        # tensor [x1] size: [1, 512, 32, 32], min: -5.017507, max: 4.429332, mean: 0.000466
        # tensor [x2] size: [1, 512, 33, 33], min: -5.017507, max: 4.429332, mean: 0.000887
        # tensor [x3] size: [1, 512, 32, 32], min: -3.52247, max: 3.504109, mean: 0.000609

        return x

def custom_avgpool2d(ctx, x):
    # ggml_shape("x1", x)
    x = ggml.ggml_replication_pad2d(ctx, x, 1, 0, 1, 0)
    # ggml_shape("x2", x)

    W = x.contents.ne[0]
    H = x.contents.ne[1]
    C = x.contents.ne[2]
    B = x.contents.ne[3]

    x = ggml.ggml_cont(ctx, ggml.ggml_reshape_3d(ctx, x, W, H, C*B));

    kernel_size = 2
    stride_size = 1
    y = ggml.ggml_pool_2d(ctx, x, ggml.GGML_OP_POOL_AVG, kernel_size, kernel_size, stride_size, stride_size, 1, 1)
    # ggml_shape("x3", y)

    y = ggml.ggml_cont(ctx, ggml.ggml_reshape_4d(ctx, y, W + 1, H + 1, C, B))

    y = ggml_nn_slice(ctx, y, 0, 1, W, 1)
    y = ggml_nn_slice(ctx, y, 1, 1, H, 1)

    # ggml_shape("x4", y)
    # x1 shape:  [32, 32, 512, 1]
    # x2 shape:  [33, 33, 512, 1]
    # x3 shape:  [34, 34, 512, 1]
    # x4 shape:  [32, 32, 512, 1]

    return y

# --------------------------------------------------------------------------------------
def test_custom_avgpool2d():
    print("test_custom_avgpool2d ...")
    print(">" * 80)
    x = torch.randn(1, 512, 32, 32)
    model = CustomAvgPool2d()
    model.eval()

    with torch.no_grad():
        torch_x = model(x)

    ctx = ggml_new_ctx(mem=1024)

    g_x = ggml_tensor(ctx, x)
    # -----------------------------------------------
    g_y = custom_avgpool2d(ctx, g_x)
    # -----------------------------------------------
    ggml_compute(ctx, g_y)

    ggml_x = torch_tensor(g_y) 
    ggml_free_ctx(ctx)

    todos.debug.output_var("torch_x", torch_x)
    todos.debug.output_var("ggm_x", ggml_x)
    todos.debug.output_var("|torch_x - ggml_x|", (torch_x - ggml_x).abs())
    print("<" * 80)

# --------------------------------------------------------------------------------------
def test_ggml_rfft2():
    print("test_ggml_rfft2 ...")
    print(">" * 80)

    x = torch.randn(1, 192, 128, 128)
    # x = torch.randn(1, 2, 6, 8)
    # x = torch.randn(1, 192, 128, 128)

    B, C, H, W = x.size()
    todos.debug.output_var("x", x)

    # tensor [input] size: [1, 192, 128, 128], min: 0.0, max: 9.555307, mean: 0.303991
    y1 =  torch.fft.rfft2(x, dim=(-2, -1), norm="backward")
    # tensor [rfft2.y1] size: [1, 192, 128, 65], min: -1139.295898, max: 75903.523438, mean: 0.563307
    todos.debug.output_var("y1", y1)

    y2 =  torch.view_as_real(y1)
    # tensor [y2] size: [1, 192, 128, 65, 2], min: -3256.635254, max: 75903.523438, mean: 0.27888
    todos.debug.output_var("y2", y2)

    # # ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
    # y2 = torch.stack((y2[..., 0], y2[..., 1]), dim=-1) # [1, 192, 128, 65, 2]
    # todos.debug.output_var("y3", y2)

    y2 = y2.permute(0, 1, 4, 2, 3).contiguous()  # [1, 192, 128, 65, 2] --> [1, 192, 2, 128, 65]
    todos.debug.output_var("y4", y2)

    y2 = y2.view((B, -1, ) + y2.size()[3:]) # [1, 384, 128, 65]
    todos.debug.output_var("y5", y2)

    torch_x = y2

    # tensor [y1] size: [1, 2, 4, 3], min: -3.007412, max: 5.854789, mean: 0.098104
    # tensor [y2] size: [1, 2, 4, 3, 2], min: -3.007412, max: 5.854789, mean: 0.142698
    # tensor [y3] size: [1, 2, 4, 3, 2], min: -3.007412, max: 5.854789, mean: 0.142698
    # tensor [y4] size: [1, 2, 2, 4, 3], min: -3.007412, max: 5.854789, mean: 0.142698
    # tensor [y5] size: [1, 4, 4, 3], min: -3.007412, max: 5.854789, mean: 0.142698


    ctx = ggml_new_ctx()

    # --------------------------------------------------------------------------------------
    gx = ggml_tensor(ctx, x)
    gy = ggml.ggml_rfft2(ctx, gx)
    igy = ggml.ggml_irfft2(ctx, gy)

    ggml_compute(ctx, igy)
    # --------------------------------------------------------------------------------------

    ggml_y = torch_tensor(gy)
    iggml_y = torch_tensor(igy)

    ggml_free_ctx(ctx)

    todos.debug.output_var("torch_x", torch_x)
    todos.debug.output_var("ggml_y", ggml_y)
    todos.debug.output_var("|torch_x - ggml_y|", (torch_x - ggml_y).abs())

    todos.debug.output_var("|x - iggml_y|", (x - iggml_y).abs())
    print("<" * 80)

# def conv_transpose2d_like_conv2d(x, in_channels, out_channels, kernel_size, stride, padding, output_padding, weight, bias):
#     # 计算上采样后的输入尺寸
#     # todos.debug.output_var("input x", x)
#     B, C, H, W = x.shape
#     H_prime = (H - 1) * stride - 2 * padding + kernel_size + output_padding
#     W_prime = (W - 1) * stride - 2 * padding + kernel_size + output_padding

#     # 上采样输入
#     x_upsampled = F.interpolate(x, size=(H_prime, W_prime), mode='bilinear', align_corners=True)
#     # x_upsampled = F.interpolate(x, size=(H_prime, W_prime), mode='nearest')

#     # 应用常规卷积
#     # conv_wq = torch.flip(weight, dims=[2, 3])
#     conv_wq = weight.transpose(0, 1)

#     output = F.conv2d(x_upsampled, conv_wq, bias, stride=1, padding=padding)

#     return output

def ggml_nn_conv_transpose2d(ctx, x, in_channels, out_channels, kernel_size, stride_size, padding, output_padding, weight, bias):
    B, C, H, W = x.size()
    weight = ggml.ggml_cast(ctx, weight, ggml.GGML_TYPE_F16)
    bias = ggml.ggml_reshape_4d(ctx, bias, 1, 1, out_channels, 1)

    y = ggml.ggml_conv_transpose_2d_p0(ctx, weight, x, stride)
    bias = ggml.ggml_repeat(ctx, bias, y)
    y = ggml_nn_slice(ctx, y, 0, 1, stride * W + 1, 1)
    y = ggml_nn_slice(ctx, y, 1, 1, stride * H + 1, 1)
    y = ggml.ggml_add(ctx, y, bias)
    return y


def test_conv_transposed2d():
    print("test_conv_transposed2d ...")
    print(">" * 80)
    in_channels = 512
    out_channels = in_channels // 2

    x = torch.randn(1, in_channels, 128, 128) # ==> [1, 256, 256, 256]
    # x = torch.randn(1, in_channels, 4, 4) # ==> [1, 256, 256, 256]

    B, C, H, W = x.size()
    todos.debug.output_var("x", x)

    model = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    # Output size=(I−1)×S −2P + K + 1

    state_dict = {}
    state_dict['weight'] = torch.randn(in_channels, out_channels, 3, 3)
    state_dict['bias'] = torch.randn(out_channels)
    # state_dict['bias'].fill_(0.0)

    model.load_state_dict(state_dict)
    model.eval()

    # like_x = conv_transpose2d_like_conv2d(x, in_channels, out_channels, 3, 2, 1, 1, state_dict['weight'], state_dict['bias'])


    with torch.no_grad():
        torch_x = model(x)
    # tensor [torch_x] size: [1, 256, 256, 256], min: -2.884916, max: 2.814204, mean: 0.000706

    ctx = ggml_new_ctx()
    g_x = ggml_tensor(ctx, x)
    g_weight = ggml_tensor(ctx, state_dict['weight'])
    # g_weight = ggml.ggml_cast(ctx, g_weight, ggml.GGML_TYPE_F16)
    g_bias = ggml_tensor(ctx, state_dict['bias'])

    # --------------------------------------------------------------------------------------
    # struct ggml_tensor * ggml_conv_transpose_2d_p0(
    #         struct ggml_context * ctx,
    #         struct ggml_tensor  * a,
    #         struct ggml_tensor  * b,
    #         int                   stride);


    # stride = 2
    # gy = ggml.ggml_conv_transpose_2d_p0(ctx, g_weight, g_x, stride)
    # gy = ggml_nn_slice(ctx, gy, 0, 1, stride * W + 1, 1)
    # gy = ggml_nn_slice(ctx, gy, 1, 1, stride * H + 1, 1)
    # # gy = ggml.ggml_upscale_ext(ctx, gy, stride * W, stride * H, out_channels, B)
    # g_bias = ggml.ggml_reshape_4d(ctx, g_bias, 1, 1, out_channels, 1)
    # g_bias = ggml.ggml_repeat(ctx, g_bias, gy)
    # gy = ggml.ggml_add(ctx, gy, g_bias)
    gy = ggml_nn_conv_transpose2d(ctx, g_x, in_channels, out_channels, kernel_size, stride_size, padding, \
            output_padding, g_weight, g_bias)


    ggml_compute(ctx, gy)
    # --------------------------------------------------------------------------------------

    ggml_y = torch_tensor(gy)
    
    ggml_free_ctx(ctx)
    print("-" * 80)
    # ggml_y = ggml_y[:, :, 1:, 1:]

    todos.debug.output_var("torch_x", torch_x)
    # todos.debug.output_var("like_x", like_x)
    todos.debug.output_var("ggml_y", ggml_y)
    todos.debug.output_var("|torch_x - ggml_y|", (torch_x - ggml_y).abs())

    # todos.debug.output_var("|torch_x - like_x|", (torch_x - like_x).abs())
    print("<" * 80)
    pdb.set_trace()




# 1)
# x = torch.randn(1, 3, 256, 256)
# y = test_reshape(x)
# print("Abs.max():", (y - x.reshape(1, 768, 16, 16)).abs().max())

# 2) OK
# test_pixel_shuffle()
# test_pixel_unshuffle()

# 3) OK
# test_ggml_mean()

# 4) OK
# test_ggml_normalize()

# 5) OK
# test_ggml_interpolate()

# 6) OK
# test_ggml_slice()

# 7) OK
# test_ggml_batch_norm2d()

# 8) OK
# test_ggml_batch_norm1d()

# 9) OK
# test_ggml_mul_mat()
# test_torch_bmm()

# 10) OK
# test_ggml_einsum()

# 11) OK
# test_ggml_position_embedding()

# 12) OK
# test_ggml_nn_softmax()
# test_ggml_multi_head_attention()

# 13) OK
# test_ggml_layer_norm_channels_first()

# 14) OK
# test_ggml_add(0.1)
# test_ggml_add(1.1)
# test_ggml_add(-0.1)

# 15) OK
# test_ggml_repeat()

# 16) OK
# test_ggml_pool_2d()

# 17) OK
# test_ggml_grid_x()

# 18) OK
# test_depthwise_conv()

# 19) OK
# test_ggml_mul_mat2()

# 20) OK
# test_ggml_nn_linear()

# 21) OK
# test_ggml_layer_norm_channels_last()

# 22) OK
# test_ggml_layer_norm()

# 23) OK
# test_custom_avgpool2d()

# 24) OK
# test_ggml_rfft2()

# 25)
test_conv_transposed2d()