import java.util.*;

import nodeRow.InputNodeRow;
import nodeRow.LeakyReLUNodeRow;
import nodeRow.NodeRow;
import nodeRow.ReLUNodeRow;

public class Client {
   public static void main(String[] args) {
      
	   Integer[] input2 = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	   Matrix<Integer> m = new Matrix<Integer>(3, 3, input2);
	   System.out.println(m);
	   Integer[] col2 = m.getCol(2, Integer.class);
	   System.out.println(Arrays.toString(col2));
	   m.setRow(0, col2);
	   System.out.println(m);
	   m.setCol(1, m.getRow(2, Integer.class));
	   System.out.println(m);
	   
	   /*
	   NodeRow[] layerPlan = { new InputNodeRow(2), new LeakyReLUNodeRow(2, 0.1), new LeakyReLUNodeRow(2, 0.1) };
	   
	   NeuralNet n = new NeuralNet(layerPlan);
	   n.initBiasRand(0, 1);
	   n.initWeightRand(0, 1);
	   
	   Double[] input = {1.0, 1.0};
	   Double[] loss = {0.5, 0.5};
	   System.out.println(Arrays.toString(n.runFunc(input)));
	   System.out.println(n);
	   n.backProp(loss, input, getNewBiasList(layerPlan), getNewWeightList(layerPlan));
	   */
      
//      Double[] array4 = new Double[6];
//      int[] array3 = new int[6];
//      for (int i = 1; i <= array3.length; i++) {
//         array3[i - 1] = i;
//         array4[i - 1] = (double) i;
//      }
//      
//      int[] array = new int[3];
//      array[0] = 2;
//      array[1] = 2;
//      array[2] = 2;
//      NeuralNet n = new NeuralNet(array3);
//      
//      n.initBiasZeros();
//      n.initWeightRand(0.0, 1.0);
//      
//      Double[] array2 = new Double[1];
//      array2[0] = 1.0;
//      List<Double[]> s = new LinkedList<Double[]>();
//      System.out.println(Arrays.toString(n.runFunc(array2, s)));
//      for (Double[] a : s) {
//         System.out.println(Arrays.toString(a));
//      }
//      
//      Double[] array5 = new Double[6];
//      Double[] array6 = new Double[6];
//      
//      Random r = new Random();
//
//      System.out.println(n);
//       for (int i = 0; i < 10; i++) {
//          s.clear();
//          array2[0] = (double) r.nextInt(100);
//          array6 = n.runFunc(array2, s);
//          for (int j = 0; j < array5.length; j++) {
//             array5[j] = array6[j] - 50;
//          }
//          n.backProp(array5, s);
//       }
//      System.out.println(n);
//      
//      n.save("test");
//      NeuralNet o = new NeuralNet("test");
//      
//      System.out.println(n);
//      System.out.println(o);
//      
//      int[] e = new int[3];
//      e[0] = 1;
//      e[1] = 0;
//      e[2] = 0;
//      
	   
	   /*
      RubiksCube f = new RubiksCube(3);      
      
      Integer[] array = new Integer[9];
      array[0] = 1;
      array[1] = 3;
      array[2] = 5;
      array[3] = 7;
      array[4] = 9;
      array[5] = 11;
      array[6] = 13;
      array[7] = 15;
      array[8] = 17;
      Matrix<Integer> a = new Matrix<Integer>(3, 3, array);
      
      System.out.println(Arrays.toString(array));
      System.out.println(Arrays.toString(a.getArray()));
      Matrix<Object> h = new Matrix<Object>(3, 3, a.getArray());
      System.out.println(a);
      System.out.println(h);
      
      System.out.println(f);
      System.out.println(Arrays.toString(f.getAllByPos()));
//       f.scrambleCube();
      f.makeMove(MoveID.R);
      f.makeMove(MoveID.U);
      f.makeMove(MoveID.RP);
      f.makeMove(MoveID.UP);
      f.makeMove(MoveID.RP);
      
      System.out.println();
      System.out.println(f);
      
      System.out.println(f.getMovesList());
      
      System.out.println(f.getOppMovesList());
      
      System.out.println(Arrays.toString(f.getAllByPiece()));
            
      
      for (Integer num : a) {
         System.out.println(num);
      }
      
      
      
      Integer[] array2 = new Integer[3];
      array2[0] = 2;
      array2[1] = 4;
      array2[2] = 6;
      System.out.println(Arrays.toString(a.multiplyVector(array2)));
      String[] array3 = new String[4];
      array3[0] = "a";
      array3[1] = "b";
      array3[2] = "c";
      array3[3] = "d";
      int[] array4 = new int[2];
      array4[0] = 2;
      array4[1] = 4;
      Matrix<String> b = new Matrix<String>(2, 2, array3);
//       System.out.println(Arrays.toString(b.multiplyVector(array4)));
      System.out.println(a.rotationalMatrix('z', 90));
      Matrix<Integer> c = a.rotationalMatrix('z', 90);
      System.out.println(a.isAllFilled());
      a.set(0, 0, null);
      System.out.println(a.isAllFilled());
      
      Number[] j = c.multiplyVector(array2);
      System.out.println(Arrays.toString(j));
      Integer[] k = (Integer[]) c.multiplyVector(array2);
      System.out.println(Arrays.toString(k)); */
      
   }
   
	private static ArrayList<Double[]> getNewBiasList(NodeRow[] layerPlan) {
		ArrayList<Double[]> biasList = new ArrayList<Double[]>();
		for (int i = 1; i < layerPlan.length; i++) {
			biasList.add(new Double[layerPlan[i].getSize()]);
			for (int j = 0; j < biasList.get(i - 1).length; j++) {
				biasList.get(i - 1)[j]=0.0;
			}
		}
		return biasList;
	}
	
	private static ArrayList<Matrix<Double>> getNewWeightList(NodeRow[] layerPlan) {
		ArrayList<Matrix<Double>> weightList = new ArrayList<Matrix<Double>>();
		for (int i = 1; i < layerPlan.length; i++) {
			weightList.add(new Matrix<Double>(layerPlan[i].getSize(), layerPlan[i - 1].getSize()));
			for (int j = 0; j < weightList.get(i - 1).getColSize(); j++) {
				for (int k = 0; k < weightList.get(i - 1).getRowSize(); k++) {
					weightList.get(i - 1).set(k, j, 0.0);
				}
			}
		}
		return weightList;
	}
}