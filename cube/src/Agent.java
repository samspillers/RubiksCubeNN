import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.Runtime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.LinkedList;
import java.util.Random;
import java.util.Scanner;
import java.text.SimpleDateFormat;  
import java.util.Date;

public class Agent {
	private static final int POSITION_REWARD = 30;
	private static final int POS_AND_OR_REWARD = 100;
	private static final int FULLY_SOLVED_REWARD = 1000;
	private static final int[] NET_LAYOUT = { 832, 422, 100, 12 };
	private static final double GAMMA = 0.9;
	private static final double EPSILON_MAX = 0.9;
	private static final double EPSILON_START = 0.01;
	private static final double EPSILON_GROWTH_RATE = 0.005;  // Increase in epsilon / 100 iterations
	private static final int REPLAYS_PER_100 = 45;

	public static void main(String[] args) {
		InputStream stream = System.in;
		Scanner reader = new Scanner(stream);
		System.out.println("Enter load file name:");
		String fileName = reader.next();
	    SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
		
		NeuralNet net;
		File tempFile = new File(fileName + ".nn");
		if (tempFile.exists()) {
			System.out.println("File found");
			net = new NeuralNet(fileName);
		} else {
			net = new NeuralNet(NET_LAYOUT);
			net.initBiasZeros();
			net.initWeightRand(0.0, 1.0);
			System.out.println("New file made");
		}
		
		RubiksCube cube = new RubiksCube(3);
		Random r = new Random();
		List<Object[]> replays = new ArrayList<Object[]>();

		List<Double> actualFutureList = new LinkedList<Double>();
		List<Integer> rewardList = new LinkedList<Integer>();
		List<Integer> rewardDiffList = new LinkedList<Integer>();
		List<Double> expectedList = new LinkedList<Double>();
		cube.scrambleCube();
		Double epsilon = EPSILON_START;
		
		int prevReward = calculateReward(cube);

		int counter = 0;
		try {
			while (stream.available() == 0) {
				Double[] prevState = getState(cube);
				Double[] funcOut = net.runFunc(prevState);
				int chosenAction = getMaxIndex(funcOut);
				Double randFloat = r.nextDouble();
				if (randFloat > epsilon) {
					int randInt = r.nextInt(12);
					while (randInt == chosenAction) {
						randInt = r.nextInt(12);
					}
					chosenAction = randInt;
				}
				takeAction(chosenAction, cube);
				Double[] currState = getState(cube);
				int reward = calculateReward(cube);
				int rewardDiff = reward - prevReward;
				addReplay(replays, prevState.clone(), chosenAction, (double) reward, currState.clone());
//				backPropFromStateAction(net, prevState, chosenAction, (double) reward, currState);
				
				Double[] nextFuncOut = net.runFunc(currState);
				rewardList.add(reward);
				rewardDiffList.add(reward - prevReward);
				actualFutureList.add(reward - prevReward + (GAMMA * nextFuncOut[getMaxIndex(nextFuncOut)]));
				expectedList.add(funcOut[getMaxIndex(funcOut)]);
				
				prevReward = reward;
				
				if (counter != 0 && counter % 100 == 0) {
					backPropFromReplay(net, replays, r, REPLAYS_PER_100);
					System.out.println(
							"[" + formatter.format(new Date()) + "] " + counter / 100 + ": Epsilon: " + epsilon + ", Reward: " + average(rewardList) + ", Reward Difference: " + average(rewardDiffList) + ", Future Rewards (A v E): " + roundAvoid(averageDouble(actualFutureList), 5) + " v "
									+ roundAvoid(averageDouble(expectedList), 5) + ", Memory: " + Runtime.getRuntime().freeMemory());
					if (epsilon < EPSILON_MAX) {
						epsilon += EPSILON_GROWTH_RATE;
					}
					actualFutureList.clear();
					expectedList.clear();
					rewardList.clear();
					net.save(fileName);
				}

				counter++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		reader.close();
	}
	
	@SuppressWarnings("unchecked")
	private static void backPropFromReplay(NeuralNet net, List<Object[]> list, Random r, int numOfSamples) {
		int[] layerPlan = net.getLayers();
		ArrayList<Double[]> inputBiases = getNewBiasList(layerPlan);
		ArrayList<Matrix<Double>> inputWeights = getNewWeightList(layerPlan);
		
		for (int i = 0; i < numOfSamples; i++) {
			int randInt = r.nextInt(list.size());
			Double[] state1 = (Double[]) list.get(randInt)[0];
			int actionTaken = (int) list.get(randInt)[1];
			Double reward = (Double) list.get(randInt)[2];
			Double[] state2 = (Double[]) list.get(randInt)[3];
			
			getGradient(net, state1, actionTaken, reward, state2, inputBiases, inputWeights);
		}
		
		// Averages all the collected gradients

		for (int i = 0; i < inputWeights.size(); i++) {
			for (int j = 0; j < inputWeights.get(i).getCols(); j++) {
				for (int k = 0; k < inputWeights.get(i).getRows(); k++) {
					inputWeights.get(i).set(k, j, inputWeights.get(i).get(k, j) / numOfSamples);
				}
			}
			
			for (int j = 0; j < inputBiases.get(i).length; j++) {
				inputBiases.get(i)[j] = inputBiases.get(i)[j] / numOfSamples;
			}
		}
		
		net.addGrad(inputBiases, inputWeights);
	}
	
	private static void backPropFromStateAction(NeuralNet net, Double[] state1, int actionTaken, Double reward, Double[] state2) {
		int[] layerPlan = net.getLayers();
		ArrayList<Double[]> inputBiases = getNewBiasList(layerPlan);
		ArrayList<Matrix<Double>> inputWeights = getNewWeightList(layerPlan);
		getGradient(net, state1, actionTaken, reward, state2, inputBiases, inputWeights);
		net.addGrad(inputBiases, inputWeights);
	}
	
	private static ArrayList<Double[]> getNewBiasList(int[] layerPlan) {
		ArrayList<Double[]> biasList = new ArrayList<Double[]>();
		for (int i = 1; i < layerPlan.length; i++) {
			biasList.add(new Double[layerPlan[i]]);
			for (int j = 0; j < biasList.get(i - 1).length; j++) {
				biasList.get(i - 1)[j]=0.0;
			}
		}
		return biasList;
	}
	
	private static ArrayList<Matrix<Double>> getNewWeightList(int[] layerPlan) {
		ArrayList<Matrix<Double>> weightList = new ArrayList<Matrix<Double>>();
		for (int i = 1; i < layerPlan.length; i++) {
			weightList.add(new Matrix<Double>(layerPlan[i], layerPlan[i - 1]));
			for (int j = 0; j < weightList.get(i - 1).getCols(); j++) {
				for (int k = 0; k < weightList.get(i - 1).getRows(); k++) {
					weightList.get(i - 1).set(k, j, 0.0);
				}
			}
		}
		return weightList;
	}
	
	private static void getGradient(NeuralNet net, Double[] state1, int actionTaken, Double reward, Double[] state2, ArrayList<Double[]> biasGrad, ArrayList<Matrix<Double>> weightGrad) {
		Double[] state1Out = net.runFunc(state1);
		Double[] state2Out = net.runFunc(state2);
		double actual = reward + GAMMA * state2Out[getMaxIndex(state2Out)];
		System.out.println("funcOut: " + Arrays.toString(state1Out));
		System.out.println("expected: " + state1Out[actionTaken] + ", actual: " + actual);
		System.out.println("loss array: " + Arrays.toString(getLossArray(actionTaken, state1Out[actionTaken] - actual)));
		
		net.backProp(getLossArray(actionTaken, state1Out[actionTaken] - actual), state1, biasGrad, weightGrad);
	}
	
	private static void addReplay(List<Object[]> list, Double[] state1, int actionTaken, Double reward, Double[] state2) {
		Object[] array = new Object[4];
		array[0] = state1;
		array[1] = actionTaken;
		array[2] = reward;
		array[3] = state2;
		list.add(array);
	}
	
	private static double averageDouble(List<Double> list) {
		double sum = 0;
		for (int i = 0; i < list.size(); i++) {
			sum += list.get(i);
		}
		return sum / list.size();
	}

	private static int average(List<Integer> list) {
		int sum = 0;
		for (int i = 0; i < list.size(); i++) {
			sum += list.get(i);
		}
		return sum / list.size();
	}
	
	private static double roundAvoid(double value, int places) {
	    double scale = Math.pow(10, places);
	    return Math.round(value * scale) / scale;
	}

	private static Double[] getLossArray(int actionIndex, Double difference) {
		if (actionIndex >= 12) {
			throw new IllegalArgumentException();
		}

		Double[] output = new Double[12];
		for (int i = 0; i < 12; i++) {
			if (i == actionIndex) {
				output[i] = difference;
			} else {
				output[i] = 0.0;
			}
		}
		return output;
	}

	private static Double[] getState(RubiksCube cube) {
		Double[] output = new Double[832];
		int[] positionArray = cube.getAllByPiece();

		for (int i = 0; i < positionArray.length; i++) {
			for (int j = 0; j < 26; j++) {
				if (j == positionArray[i]) {
					output[i * 32 + j] = 1.0;
				} else {
					output[i * 32 + j] = 0.0;
				}
			}

			int oriState = getOriState(cube.getById(i).getOrientation());
			for (int j = 0; j < 6; j++) {
				if (j == oriState) {
					output[i * 32 + j + 26] = 1.0;
				} else {
					output[i * 32 + j + 26] = 0.0;
				}
			}
		}

		return output;
	}

	private static int getMaxIndex(Double[] array) {
		int maxIndex = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] > array[maxIndex]) {
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	private static int getOriState(int[] ori) {
		if (ori.length != 3) {
			throw new IllegalArgumentException();
		}

		int[] xPos = { 1, 0, 0 };
		int[] xNeg = { -1, 0, 0 };
		int[] yPos = { 0, 1, 0 };
		int[] yNeg = { 0, -1, 0 };
		int[] zPos = { 0, 0, 1 };
		int[] zNeg = { 0, 0, -1 };

		if (Arrays.equals(ori, xPos)) {
			return 0;
		} else if (Arrays.equals(ori, xNeg)) {
			return 1;
		} else if (Arrays.equals(ori, yPos)) {
			return 2;
		} else if (Arrays.equals(ori, yNeg)) {
			return 3;
		} else if (Arrays.equals(ori, zPos)) {
			return 4;
		} else if (Arrays.equals(ori, zNeg)) {
			return 5;
		} else {
			throw new IllegalArgumentException();
		}
	}

	private static int calculateReward(RubiksCube cube) {
		int rewardSum = 0;

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					if (i != 1 || j != 1 || k != 1) {
						if (cube.pieceCorrectPos(i, j, k)) {
							if (cube.pieceIsSolved(i, j, k)) {
								rewardSum += POS_AND_OR_REWARD;
							} else {
								rewardSum += POSITION_REWARD;
							}
						}
					}
				}
			}
		}

		if (cube.isSolved()) {
			rewardSum += FULLY_SOLVED_REWARD;
		}
		return rewardSum;
	}

	private static void takeAction(int actionIndex, RubiksCube cube) {
		switch (actionIndex) {
		case 0:
			cube.makeMove(MoveID.F);
			break;
		case 1:
			cube.makeMove(MoveID.FP);
			break;
		case 2:
			cube.makeMove(MoveID.B);
			break;
		case 3:
			cube.makeMove(MoveID.BP);
			break;
		case 4:
			cube.makeMove(MoveID.U);
			break;
		case 5:
			cube.makeMove(MoveID.UP);
			break;
		case 6:
			cube.makeMove(MoveID.D);
			break;
		case 7:
			cube.makeMove(MoveID.DP);
			break;
		case 8:
			cube.makeMove(MoveID.L);
			break;
		case 9:
			cube.makeMove(MoveID.LP);
			break;
		case 10:
			cube.makeMove(MoveID.R);
			break;
		case 11:
			cube.makeMove(MoveID.RP);
			break;
		}
	}

}