package nodeRow;

public interface NodeRow {
	public Double[] runActivationFunc(Double[] array);
	
	public Double[] runActivationFuncDerivative(Double[] array);
	
	public int getSize();
}