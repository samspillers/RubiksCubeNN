import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;

public class NeuralNet {
	private ArrayList<Double[]> biasList;
	private ArrayList<Matrix<Double>> weightList;
	private int[] layerPlan;
	public static final double LEAKY_RELU_SLOPE = 0.01;  // Slope of the line to the left of x = 0

	// Give array of ints. Each entry is a layer, each int is the number of nodes in
	// that lay. Include
	// input and output layers
	public NeuralNet(int[] layerPlan) {
		biasList = new ArrayList<Double[]>();
		weightList = new ArrayList<Matrix<Double>>();
		this.layerPlan = layerPlan;

		for (int i = 1; i < layerPlan.length; i++) {
			biasList.add(new Double[layerPlan[i]]);
			weightList.add(new Matrix<Double>(layerPlan[i], layerPlan[i - 1]));
		}
	}

	// Used to load a previously made NN. String must be the name of a correctly
	// formatted save file
	public NeuralNet(String fileName) {
		biasList = new ArrayList<Double[]>();
		weightList = new ArrayList<Matrix<Double>>();

		load(fileName);
	}

	public void initWeightRand(double minWeight, double maxWeight) {
		Random r = new Random();

		for (Matrix<Double> matrix : weightList) {
			for (int i = 0; i < matrix.getRows(); i++) {
				for (int j = 0; j < matrix.getCols(); j++) {
					matrix.set(i, j, r.nextDouble() * (maxWeight - minWeight) + minWeight);
				}
			}
		}
	}

	public void initBiasRand(double minBias, double maxBias) {
		Random r = new Random();
		for (Double[] array : biasList) {
			for (int i = 0; i < array.length; i++) {
				array[i] = r.nextDouble() * (maxBias - minBias) + minBias;
			}
		}
	}

	public void initWeightZeros() {
		for (Matrix<Double> matrix : weightList) {
			for (int i = 0; i < matrix.getRows(); i++) {
				for (int j = 0; j < matrix.getCols(); j++) {
					matrix.set(i, j, 0.0);
				}
			}
		}
	}

	public void initBiasZeros() {
		for (Double[] array : biasList) {
			for (int i = 0; i < array.length; i++) {
				array[i] = 0.0;
			}
		}
	}

	// Runs NN function. Inputs and outputs can be beyond 0-1, your job to check
	// that if that's not what you want
	public List<Double[]> getActivations(Double[] input) {
		List<Double[]> activations = new ArrayList<Double[]>();
		getActivations(input, activations, 0);
//		for (int i = 0; i < activations.size(); i++) {
//			System.out.println(Arrays.toString(activations.get(i)));
//		}
		return activations;
	}

	// Runs NN function recursively, one recursion for each layer. Basically:
	// runFunc(ReLU(addVectors(weightArray * inputArray,biasArray)))
	private void getActivations(Double[] input, List<Double[]> activations, int index) {
		activations.add(input.clone());
		if (index != weightList.size()) {
			if (input.length != weightList.get(index).getCols()) {
				throw new IllegalArgumentException();
			}
			getActivations(runReLU(
					subtractArrays((Double[]) weightList.get(index).multiplyVector(input), biasList.get(index))),
					activations, index + 1);
		}
	}

	// Runs NN function. Inputs and outputs can be beyond 0-1, your job to check
	// that if that's not what you want
	public Double[] runFunc(Double[] input) {
		return runFunc(input, 0);
	}

	// Runs NN function recursively, one recursion for each layer. Basically:
	// runFunc(ReLU(addVectors(weightArray * inputArray,biasArray)))
	private Double[] runFunc(Double[] input, int index) {
		if (index == weightList.size()) {
			return input.clone();
		} else {
			if (input.length != weightList.get(index).getCols()) {
				throw new IllegalArgumentException();
			}
			return runFunc(runReLU(
					subtractArrays((Double[]) weightList.get(index).multiplyVector(input), biasList.get(index))),
					index + 1);
		}
	}

	// Give array of loss for each final node, and the activations of each layer of
	// the corresponding loss run
	public void backProp(Double[] lossArray, Double[] state, ArrayList<Double[]> biasGrad, ArrayList<Matrix<Double>> weightGrad) {
		backProp(lossArray, getActivations(state), biasGrad, weightGrad, 0);
	}

	// Works back to front, changes weights and bias according to the loss array and
	// previous activations
	private void backProp(Double[] lossArray, List<Double[]> activations, ArrayList<Double[]> biasGrad, ArrayList<Matrix<Double>> weightGrad, int index) {
		if (index < weightList.size()) {
			if (lossArray.length != weightList.get(weightList.size() - index - 1).getRows()) {
				throw new IllegalArgumentException();
			}
			int listInd = weightList.size() - index - 1;
			// Since activationGradient depends on weights, if the weights are changed after
			// the activation gradient is calculated, nothing goes wrong(?)
			Double[] activationGradient = new Double[weightList.get(listInd).getCols()];
			for (int i = 0; i < activationGradient.length; i++) {
				activationGradient[i] = 0.0;
			}

			// i is for every node in the backer layer, i.e. the older layer. j is for every
			// node in the fronter layer, the new layer
			for (int i = 0; i < weightList.get(listInd).getRows(); i++) {
				for (int j = 0; j < weightList.get(listInd).getCols(); j++) {
					activationGradient[j] += weightList.get(listInd).get(i, j)
							* derivativeReLU(activations.get(listInd + 1)[i]) * lossArray[i];
					weightGrad.get(listInd).set(i, j, weightGrad.get(listInd).get(i, j) + activations.get(listInd)[j]
							* derivativeReLU(activations.get(listInd + 1)[i]) * lossArray[i]);
				}
				// Biases affect nothing and can be changed immediately
				biasGrad.get(listInd)[i] += derivativeReLU(activations.get(listInd + 1)[i]) * lossArray[i];
			}
			backProp(activationGradient, activations, biasGrad, weightGrad, index + 1);
		}
	}

	// For updating the NN after taking back prop averages
	public void addGrad(ArrayList<Double[]> biasGrad, ArrayList<Matrix<Double>> weightGrad) {
		for (int i = 0; i < weightList.size(); i++) {
			for (int j = 0; j < weightList.get(i).getCols(); j++) {
				for (int k = 0; k < weightList.get(i).getRows(); k++) {
					weightList.get(i).set(k, j, weightList.get(i).get(k, j) + weightGrad.get(i).get(k, j));
				}
			}

			for (int j = 0; j < biasList.get(i).length; j++) {
				biasList.get(i)[j] += biasGrad.get(i)[j];
			}
		}
	}

	private Double[] addArrays(Double[] array1, Double[] array2) {
		if (array1.length != array2.length) {
			throw new IllegalArgumentException();
		}
		Double[] output = new Double[array1.length];
		for (int i = 0; i < array1.length; i++) {
			output[i] = array1[i] + array2[i];
		}
		return output;
	}

	// Subtracts second array from first, i.e. array1 - array2
	private Double[] subtractArrays(Double[] array1, Double[] array2) {
		if (array1.length != array2.length) {
			throw new IllegalArgumentException();
		}
		Double[] output = new Double[array1.length];
		for (int i = 0; i < array1.length; i++) {
			output[i] = array1[i] - array2[i];
		}
		return output;
	}

	private Double[] runReLU(Double[] array) {
		Double[] output = new Double[array.length];
		for (int i = 0; i < array.length; i++) {
			if (array[i] > 0) {
				output[i] = array[i];
			} else {
				output[i] = 0.0;
			}
		}
		return output;
	}
	
	private Double[] runLeakyReLU(Double[] array) {
		Double[] output = new Double[array.length];
		for (int i = 0; i < array.length; i++) {
			if (array[i] > 0) {
				output[i] = array[i];
			} else {
				output[i] = array[i] * LEAKY_RELU_SLOPE;
			}
		}
		return output;
	}

	// Returns ReLU'(z(x)), given ReLU(z(x))
	private Double derivativeReLU(Double activation) {
		if (activation != 0.0) {
			return 1.0;
		} else {
			return 0.0;
		}
	}
	
	private Double derivativeLeakyReLU(Double activation) {
		if (activation != 0.0) {
			return 1.0;
		} else {
			return LEAKY_RELU_SLOPE;
		}
	}

	public String toString() {
		String output = "{";
		for (int i = 0; i < weightList.size(); i++) {
			output = output + weightList.get(i).toString() + System.lineSeparator() + Arrays.toString(biasList.get(i));
			if (i < weightList.size() - 1) {
				output = output + System.lineSeparator() + System.lineSeparator();
			}
		}

		return output + "}";
	}

	public void save(String fileName) {
		try {
			File file = new File(fileName + ".nn");
			file.delete();
			FileOutputStream fos = new FileOutputStream(file);
			DataOutputStream outStream = new DataOutputStream(new BufferedOutputStream(fos));

			outStream.writeInt(weightList.get(0).getCols());
			for (int i = 0; i < weightList.size(); i++) {
				outStream.writeInt(weightList.get(i).getRows());
			}
			outStream.writeInt(0);

			for (int i = 0; i < weightList.size(); i++) {
				for (int j = 0; j < weightList.get(i).getCols(); j++) {
					for (int k = 0; k < weightList.get(i).getRows(); k++) {
						outStream.writeDouble(weightList.get(i).get(k, j));
					}
				}
				for (int j = 0; j < biasList.get(i).length; j++) {
					outStream.writeDouble(biasList.get(i)[j]);
				}
			}

			outStream.close();
		} catch (Exception FileNotFoundExpcetion) {
			throw new IllegalArgumentException();
		}
	}

	private void load(String fileName) {
		if (weightList.size() != 0 || biasList.size() != 0) {
			throw new IllegalArgumentException();
		}

		try {
			FileInputStream fis = new FileInputStream(new File(fileName + ".nn"));
			DataInputStream reader = new DataInputStream(fis);

			int prevLayer = reader.readInt();
			int layerCounter = 1;
			for (int nextLayer = reader.readInt(); nextLayer != 0; nextLayer = reader.readInt()) {
				weightList.add(new Matrix<Double>(nextLayer, prevLayer));
				biasList.add(new Double[nextLayer]);

				layerCounter++;
				prevLayer = nextLayer;
			}
			
			
			int[] layerPlan = new int[layerCounter];
			layerPlan[0] = weightList.get(0).getCols();
			for (int i = 1; i < layerCounter; i++) {
				layerPlan[i] =  weightList.get(i - 1).getRows();
			}
			this.layerPlan = layerPlan;

			for (int i = 0; i < weightList.size(); i++) {
				for (int j = 0; j < weightList.get(i).getCols(); j++) {
					for (int k = 0; k < weightList.get(i).getRows(); k++) {
						weightList.get(i).set(k, j, reader.readDouble());
					}
				}
				for (int j = 0; j < biasList.get(i).length; j++) {
					biasList.get(i)[j] = reader.readDouble();
				}
			}
			reader.close();

		} catch (Exception FileNotFoundExpcetion) {
 
		}
	}
	
	public int[] getLayers() {
		return layerPlan.clone();
	}
}