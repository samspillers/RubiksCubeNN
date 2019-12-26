package nodeRow;

public class ReLUNodeRow implements NodeRow {
	
	private int size;
	
	public ReLUNodeRow(int rowSize) {
		size = rowSize;
	}
	
	public Double[] runActivationFunc(Double[] input) {
		if (input.length != size) {
			throw new IllegalArgumentException();
		}
		Double[] output = new Double[input.length];
		for (int i = 0; i < input.length; i++) {
			if (input[i] > 0) {
				output[i] = input[i];
			} else {
				output[i] = 0.0;
			}
		}
		return output;
	}
	
	public Double[] runActivationFuncDerivative(Double[] input) {
		if (input.length != size) {
			throw new IllegalArgumentException();
		}
		Double[] output = new Double[input.length];
		for (int i = 0; i < input.length; i++) {
			if (input[1] >= 0.0) {
				output[i] = 1.0;
			} else {
				output[i] = 0.0;
			}
		}
		return output;
	}
	
	public int getSize() {
		return size;
	}
}
