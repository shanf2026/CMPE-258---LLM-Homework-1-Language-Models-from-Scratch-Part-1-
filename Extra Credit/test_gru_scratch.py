import torch
import torch.nn as nn

from nanolm.models.rnn import GRUScratch


def test_gru_matches_pytorch():
    torch.manual_seed(1234)

    batch = 3
    seq_len = 7
    input_dim = 5
    hidden_dim = 11

    x = torch.randn(batch, seq_len, input_dim)
    h0 = torch.randn(batch, hidden_dim)

    reference = nn.GRU(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=1,
        batch_first=True,
        dropout=0.0,
    )

    scratch = GRUScratch(input_dim, hidden_dim)

    # Copy PyTorch's parameters into the scratch implementation.
    with torch.no_grad():
        scratch.W_ih.weight.copy_(reference.weight_ih_l0)
        scratch.W_ih.bias.copy_(reference.bias_ih_l0)

        scratch.W_hh.weight.copy_(reference.weight_hh_l0)
        scratch.W_hh.bias.copy_(reference.bias_hh_l0)

    expected_output, expected_hidden = reference(
        x,
        h0.unsqueeze(0),
    )

    actual_output = scratch(x, h0)
    actual_hidden = actual_output[:, -1].unsqueeze(0)

    output_error = (actual_output - expected_output).abs().max().item()
    hidden_error = (actual_hidden - expected_hidden).abs().max().item()

    print("maximum output error:", output_error)
    print("maximum hidden error:", hidden_error)

    assert torch.allclose(
        actual_output,
        expected_output,
        atol=1e-6,
        rtol=1e-5,
    )

    assert torch.allclose(
        actual_hidden,
        expected_hidden,
        atol=1e-6,
        rtol=1e-5,
    )

def test_gru_gradients_match():
    torch.manual_seed(1234)

    x1 = torch.randn(2, 4, 5, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    reference = nn.GRU(5, 11, batch_first=True)
    scratch = GRUScratch(5, 11)

    with torch.no_grad():
        scratch.W_ih.weight.copy_(reference.weight_ih_l0)
        scratch.W_ih.bias.copy_(reference.bias_ih_l0)
        scratch.W_hh.weight.copy_(reference.weight_hh_l0)
        scratch.W_hh.bias.copy_(reference.bias_hh_l0)

    y_reference, _ = reference(x1)
    y_scratch = scratch(x2)

    loss_reference = y_reference.square().sum()
    loss_scratch = y_scratch.square().sum()

    loss_reference.backward()
    loss_scratch.backward()

    input_error = (x1.grad - x2.grad).abs().max().item()

    print("maximum input-gradient error:", input_error)

    assert torch.allclose(
        x1.grad,
        x2.grad,
        atol=1e-6,
        rtol=1e-5,
    )