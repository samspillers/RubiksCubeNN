package nodeRow;

public class LeakyReLUNodeRow implements NodeRow {
	
	private int size;
	private double leakySlope;
	
	public LeakyReLUNodeRow(int rowSize, double slope) {
		size = rowSize;
		leakySlope = slope;
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
				output[i] = input[i] * leakySlope;
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
				output[i] = leakySlope;
			}
		}
		return output;
	}
	
	public int getSize() {
		return size;
	}
}
