import java.util.ArrayList;

public interface NueralNetInterface {
	public void initWeightRand(double minWeight, double maxWeight);
	
	public void initBiasRand(double minBias, double maxBias);
	
	public void initWeightZeros();
	
	public void initBiasZeros();
	
	public Double[] runFunc(Double[] input);
	
	public void backProp(Double[] lossArray, Double[] state, ArrayList<Double[]> biasGrad, ArrayList<Matrix<Double>> weightGrad);
	
	public void save(String fileName);
}
