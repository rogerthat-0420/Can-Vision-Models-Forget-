from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import argparse

app = Flask(__name__)

# These will be filled at runtime
regular_tensor = None
quantized_tensor = None
sorrted_channels = None

def load_tensor(path):
    tensor = torch.load(path)
    if tensor.ndim == 4:
        tensor = tensor[0]  # shape: (C, H, W)
    return tensor.cpu().numpy()

def compute_impact(r, q):
    impact = [np.sum(np.abs(r[i] - q[i])) for i in range(r.shape[0])]
    return sorted([(i, val) for i, val in enumerate(impact)], key=lambda x: -x[1])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/channel-list')
def channel_list():
    return jsonify([
        {'index': i, 'impact': float(impact)}
        for i, impact in sorted_channels
    ])

@app.route('/feature-maps')
def get_maps():
    ch = int(request.args.get('channel', 0))
    reg = regular_tensor[ch]
    quant = quantized_tensor[ch]
    diff = np.abs(reg - quant)
    return jsonify({
        'regular': reg.tolist(),
        'quantized': quant.tolist(),
        'difference_abs': diff.tolist(),
        'difference_rel': (diff / np.abs(reg)).tolist(),
    })

def main():
    global regular_tensor, quantized_tensor, sorted_channels

    parser = argparse.ArgumentParser(description="Visualize feature maps from two tensors")
    parser.add_argument('--regular', type=str, required=True, help='Path to regular tensor (e.g., regular.pt)')
    parser.add_argument('--quantized', type=str, required=True, help='Path to quantized tensor (e.g., quantized.pt)')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the app on')
    args = parser.parse_args()

    regular_tensor = load_tensor(args.regular)
    quantized_tensor = load_tensor(args.quantized)

    if regular_tensor.shape != quantized_tensor.shape:
        raise ValueError("Tensors must have the same shape")
    
    sorted_channels = compute_impact(regular_tensor, quantized_tensor)

    app.run(debug=True, port=args.port)

if __name__ == '__main__':
    main()
