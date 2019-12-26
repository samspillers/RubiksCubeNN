package nodeRow;

public class SigmoidNodeRow implements NodeRow {

	private int size;
	
	public SigmoidNodeRow(int size) {
		this.size = size;
	}
	
	public Double[] runActivationFunc(Double[] input) {
		if (input.length != size) {
			throw new IllegalArgumentException();
		}
		Double[] output = new Double[input.length];
		for (int i = 0; i < input.length; i++) {
			output[i] = 1 / (1 + Math.exp(-output[i]));
		}
		return output;
	}

	public Double[] runActivationFuncDerivative(Double[] input) {
		if (input.length != size) {
			throw new IllegalArgumentException();
		}
		Double[] output = new Double[input.length];
		for (int i = 0; i < input.length; i++) {
			double sigmoidOut = 1 / (1 + Math.exp(-output[i]));
			output[i] = sigmoidOut * (1 - sigmoidOut);
		}
		return output;
	}

	public int getSize() {
		return size;
	}
}
