import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import nodeRow.*;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.BufferedOutputStream;
import java.io.File;

public class NeuralNet {
	private ArrayList<Double[]> biasList;
	private ArrayList<Matrix<Double>> weightList;
	private NodeRow[] layerPlan;
	public static final double LEAKY_RELU_SLOPE = 0.01;  // Slope of the line to the left of x = 0
	private static Random r = new Random();
	private final int SIZE;

	// Give array of ints. Each entry is a layer, each int is the number of nodes in
	// that lay. Include
	// input and output layers
	public NeuralNet(NodeRow[] layerPlan) {
		biasList = new ArrayList<Double[]>();
		weightList = new ArrayList<Matrix<Double>>();
		this.layerPlan = layerPlan;

		for (int i = 1; i < layerPlan.length; i++) {
			biasList.add(new Double[layerPlan[i].getSize()]);
			weightList.add(new Matrix<Double>(layerPlan[i].getSize(), layerPlan[i - 1].getSize()));
		}
		
		SIZE = layerPlan.length;
	}

	// Used to load a previously made NN. String must be the name of a correctly
	// formatted save file
	public NeuralNet(String fileName) {
		biasList = new ArrayList<Double[]>();
		weightList = new ArrayList<Matrix<Double>>();

		load(fileName);
		
		SIZE = layerPlan.length;
	}

	public void initWeightRand(double minWeight, double maxWeight) {
		for (Matrix<Double> matrix : weightList) {
			for (int i = 0; i < matrix.getRowSize(); i++) {
				for (int j = 0; j < matrix.getColSize(); j++) {
					matrix.set(i, j, r.nextDouble() * (maxWeight - minWeight) + minWeight);
				}
			}
		}
	}

	public void initBiasRand(double minBias, double maxBias) {
		for (Double[] array : biasList) {
			for (int i = 0; i < array.length; i++) {
				array[i] = r.nextDouble() * (maxBias - minBias) + minBias;
			}
		}
	}

	public void initWeightZeros() {
		for (Matrix<Double> matrix : weightList) {
			for (int i = 0; i < matrix.getRowSize(); i++) {
				for (int j = 0; j < matrix.getColSize(); j++) {
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
		System.out.println("Weight index, from input to out: " + index);
		activations.add(input.clone());
		if (index != weightList.size()) {
			if (input.length != weightList.get(index).getColSize()) {
				throw new IllegalArgumentException();
			}
			getActivations(layerPlan[index + 1].runActivationFunc(
					subArr((Double[]) weightList.get(index).multVec(input), biasList.get(index))),
					activations, index + 1);
		}
	}
	
	// Same as getActivations, but only returns last result
	public Double[] runFunc(Double[] input) {
		return getActivations(input).get(weightList.size());
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
			if (lossArray.length != weightList.get(weightList.size() - index - 1).getRowSize()) {
				throw new IllegalArgumentException();
			}
			int listInd = weightList.size() - index - 1;
			System.out.println("listInd: " + listInd);
			System.out.println("weightList.size(): " + weightList.size());
			// Since activationGradient depends on weights, if the weights are changed after
			// the activation gradient is calculated, nothing goes wrong(?)
			Double[] activationGradient = new Double[weightList.get(listInd).getColSize()];
			// Init array to zeros
			for (int i = 0; i < activationGradient.length; i++) {
				activationGradient[i] = 0.0;
			}

			// i is for every node in the backer layer, i.e. the older layer. j is for every
			// node in the fronter layer, the new layer
			for (int i = 0; i < weightList.get(listInd).getRowSize(); i++) {
				System.out.println("i: " + i);
				
				addArr(  // Indented for clarity
						activationGradient,
						mulArr(
								weightGrad.get(listInd).getRow(i, Double.class),  // TODO: maybe change getRow name to col, whichever is correct
								layerPlan[index + 1].runActivationFunc(activations.get(listInd + 1)),
								lossArray));
				weightGrad.get(listInd).setRow(i, 
						mulArr(
								weightGrad.get(listInd).getRow(i, Double.class),
								layerPlan[index + 1].runActivationFunc(activations.get(listInd + 1)),
								lossArray));
				
				for (int j = 0; j < weightList.get(listInd).getColSize(); j++) {
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
			for (int j = 0; j < weightList.get(i).getColSize(); j++) {
				for (int k = 0; k < weightList.get(i).getRowSize(); k++) {
					weightList.get(i).set(k, j, weightList.get(i).get(k, j) + weightGrad.get(i).get(k, j));
				}
			}

			for (int j = 0; j < biasList.get(i).length; j++) {
				biasList.get(i)[j] += biasGrad.get(i)[j];
			}
		}
	}

	@SuppressWarnings("unused")
	private Double[] addArr(Double[] array1, Double[] array2) {
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
	private Double[] subArr(Double[] array1, Double[] array2) {
		if (array1.length != array2.length) {
			throw new IllegalArgumentException();
		}
		Double[] output = new Double[array1.length];
		for (int i = 0; i < array1.length; i++) {
			output[i] = array1[i] - array2[i];
		}
		return output;
	}
	
	// Multiplies the array elements, i.e. array1 * array2
	private Double[] mulArr(Double[]... arrays) {
		int size = arrays[0].length;
		for (int i = 1; i < arrays.length; i++) {
			if (arrays[i].length != size) {
				throw new IllegalArgumentException();
			}
		}

		Double[] output = arrays[0].clone();
		for (int i = 1; i < arrays.length; i++) {
			for (int j = 0; j < size; j++) {
				output[j] = output[j] * arrays[i][j];
			}
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
		if (activation > 0.0) {
			return 1.0;
		} else {
			return 0.0;
		}
	}
	
	private Double derivativeLeakyReLU(Double activation) {
		if (activation >= 0.0) {
			return 1.0;
		} else {
			return LEAKY_RELU_SLOPE;
		}
	}

	public String toString() {
		String output = "[";
		for (int i = 0; i < weightList.size(); i++) {
			output = output + "Weights, index " + i + ": " + System.lineSeparator() + weightList.get(i).toString() + System.lineSeparator() + "Biases, index " + i + ": " + System.lineSeparator() + Arrays.toString(biasList.get(i));
			if (i < weightList.size() - 1) {
				output = output + System.lineSeparator() + System.lineSeparator();
			}
		}
		return output + "]";
	}

	public void save(String fileName) {
		try {
			File file = new File(fileName + ".nn");
			file.delete();
			FileOutputStream fos = new FileOutputStream(file);
			DataOutputStream outStream = new DataOutputStream(new BufferedOutputStream(fos));

			outStream.writeInt(weightList.get(0).getColSize());
			for (int i = 0; i < weightList.size(); i++) {
				outStream.writeInt(weightList.get(i).getRowSize());
			}
			outStream.writeInt(0);

			for (int i = 0; i < weightList.size(); i++) {
				for (int j = 0; j < weightList.get(i).getColSize(); j++) {
					for (int k = 0; k < weightList.get(i).getRowSize(); k++) {
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
			
			
			NodeRow[] layerPlan = new NodeRow[layerCounter];
			layerPlan[0] = new ReLUNodeRow(weightList.get(0).getColSize());
			for (int i = 1; i < layerCounter; i++) {
				layerPlan[i] =  new ReLUNodeRow(weightList.get(i - 1).getRowSize());
			}
			this.layerPlan = layerPlan;

			for (int i = 0; i < weightList.size(); i++) {
				for (int j = 0; j < weightList.get(i).getColSize(); j++) {
					for (int k = 0; k < weightList.get(i).getRowSize(); k++) {
						weightList.get(i).set(k, j, reader.readDouble());
					}
				}
				for (int j = 0; j < biasList.get(i).length; j++) {
					biasList.get(i)[j] = reader.readDouble();
				}
			}
			reader.close();

		} catch (Exception FileNotFoundExpcetion) {
			FileNotFoundExpcetion.printStackTrace();
		}
	}
	
	public NodeRow[] getLayers() {
		return layerPlan.clone();
	}
}