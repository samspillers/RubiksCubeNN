package nodeRow;

public class InputNodeRow implements NodeRow {

	private int size;
	
	public InputNodeRow(int size) {
		this.size = size;
	}
	
	public Double[] runActivationFunc(Double[] array) {
		throw new UnsupportedOperationException();
	}

	public Double[] runActivationFuncDerivative(Double[] array) {
		throw new UnsupportedOperationException();
	}

	public int getSize() {
		return size;
	}
	
}
