import java.util.*;
import java.lang.reflect.Array;
import java.lang.IllegalArgumentException;
import java.lang.Integer;

/* A matrix class. Includes some matrix operations.
 */

@SuppressWarnings("unchecked")  // Don't worry, it's safe
// This class stores the E objects in an Object array, so a lot of unchecked casting is done
public class Matrix <E> implements Iterable<E> {
   /*
   Matrix elements are store very similarly to an ArrayList
   Matrix elements are held in one flattened array. See diagram below for how matrix elements are stored in array
    
   Formatting of matrix:
   Array: 0 1 2 3 4 ...
   Matrix:
   0 4 8  12
   1 5 9  13
   2 6 10 14
   3 7 11 15
   
   Col and row numbers start at 0
   Indexes are given (row #, col #)
   
   // Command to convert from matrix indeces to array index
   Matrix[rowDex, colDex] = Array[rowDex + colDex * rows]
   */

	// I don't recall why these are protected, I might've copied this part from ArrayList
   protected int rows;
   protected int cols;
   protected Object[] data;  // Holds elements in a single stretched out array
   
   
   // Makes an empty matrix of the given size
   public Matrix(int rows, int cols) {
      this(rows, cols, (E[]) new Object[rows * cols]);
   }
   
   // Matrix clone constructor
   public Matrix(Matrix<E> m) {
      data = m.data.clone();
      this.rows = m.rows;
      this.cols = m.cols;
   }
   
   // Makes a matrix of the given size with data in the array format described above. Throws exception if array doesn't match given matrix size.
   public Matrix(int rows, int cols, E[] data) {
      this.data = new Object[rows * cols];
      if (data.length != rows * cols) {
         throw new IllegalArgumentException();
      }
      
      this.rows = rows;
      this.cols = cols;
      
      for (int i = 0; i < data.length; i++) {
         this.data[i] = data[i];
      }
   }
   
   public int getRows() {
      return rows;
   }
   
   public int getCols() {
      return cols;
   }
   
   public Object[] getArray() {
      return data.clone();
   }
   
   // Returns true if given cell is not null
   public boolean isFilled(int rowDex, int colDex) {
      return get(rowDex, colDex) != null;
   }
   
   // Returns true if all cells aren't null
   public boolean isAllFilled() {
      boolean output = true;
      for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
            output = output && isFilled(i, j);
         }
      }
      return output;
   }
   
   // Clones matrix
   public Matrix<E> clone() {
      return new Matrix<E>(rows, cols, (E[]) data.clone());
   }
   
   // Clears matrix
   public void clear() {
      data = new Object[rows * cols];
   }
   
   // Sets specified cell to given entry
   public void set (int rowDex, int colDex, E entry) {
      if (rowDex + colDex * rows >= rows * cols) {
         throw new IllegalArgumentException();
      }
      data[rowDex + colDex * rows] = entry;
   }
   
   // Returns element in given cell
   public E get (int rowDex, int colDex) {
      if (rowDex + colDex * rows >= rows * cols) {
         throw new IllegalArgumentException();
      }
      return (E) data[rowDex + colDex * rows];
   }
   
   // Converts matrix to string with two spaces between elements
   public String toString() {
      String output = "";
      for (int i = 0; i < rows; i++) {
         for (int j = 0; j < cols; j++) {
            output = output + get(i,j);
            if (j < cols - 1) {
               output = output + "  ";
            }
         }
         if (i < rows - 1) {
            output = output + System.lineSeparator();
         }
      }
      return output;
   }
   
   // This returns a rotation matrix for 3d space. Multiplying vectors to these rotational matrices will rotate
   // 	 the vectors around the given axis for the given angle.
   //
   // These really shouldn't return Matrix<Integer> but instead double or long or something, but since I won't
   //    be using angles other than 90 or -90, it shouldn't matter. TODO: Fix to be a functional matrix later.
   // Returns a rotational matrix to multiply vectors on, needs axis ('x', 'y', or 'z') and angle in degrees
   public static Matrix<Integer> rotationalMatrix(char axis, float angle) {
      if (axis != 'x' && axis != 'y' && axis != 'z') {
         throw new IllegalArgumentException();
      }
      
      if (axis == 'x') {
         double[] array = {1, 0, 0,
                           0, Math.cos(Math.toRadians(angle)), Math.sin(Math.toRadians(angle)),
                           0, -Math.sin(Math.toRadians(angle)), Math.cos(Math.toRadians(angle))};
         Integer[] array2 = new Integer[9];
         for (int i = 0; i < 9; i++) {
            array2[i] = Integer.valueOf((int) Math.round(array[i]));
         }
         return new Matrix<Integer>(3, 3, array2);
      } else if (axis == 'y') {
         double[] array = {Math.cos(Math.toRadians(angle)), 0, Math.sin(Math.toRadians(angle)),
                           0, 1, 0,
                           Math.sin(Math.toRadians(angle)), 0, Math.cos(Math.toRadians(angle))};
         Integer[] array2 = new Integer[9];
         for (int i = 0; i < 9; i++) {
            array2[i] = Integer.valueOf((int) Math.round(array[i]));
         }
         return new Matrix<Integer>(3, 3, array2);
      } else {
         double[] array = {Math.cos(Math.toRadians(angle)), -Math.sin(Math.toRadians(angle)), 0,
                           Math.sin(Math.toRadians(angle)), Math.cos(Math.toRadians(angle)), 0,
                           0, 0, 1};
         
         Integer[] array2 = new Integer[9];
         for (int i = 0; i < 9; i++) {
            array2[i] = Integer.valueOf((int) Math.round(array[i]));
         }
         return new Matrix<Integer>(3, 3, array2);
      }
   }

   // Only works on square matrices
   public void rotateEntriesClock() {
      if (rows != cols) {
         throw new IllegalArgumentException();
      }
      
      for (int i = 0; i < rows / 2; i++) {  // i for each circle of the matrix
         int m = i;  // curr x pos
         int n = i;  // curr y pos
         boolean forward = true;
         boolean cont;
         Queue<E> queueLoop = new LinkedList();
         for (int j = 0; j < 2; j++) {
            cont = checkCont(forward, m, cols - 1, i);
            while (cont) {
               queueLoop.add(get(m,n));
               m = addOrSubtractOne(forward, m);
               cont = checkCont(forward, m, cols - 1, i);
            }
            cont = checkCont(forward, n, cols - 1, i);
            while (cont) {
               queueLoop.add(get(m,n));
               n = addOrSubtractOne(forward, n);
               cont = checkCont(forward, n, cols - 1, i);
            }
            forward = !forward;
         }
         for (int j = 0; j < rows - 1; j++) {
            queueLoop.add(queueLoop.remove());
         }
         for (int j = 0; j < 2; j++) {
            cont = checkCont(forward, m, cols - 1, i);
            while (cont) {
               set(m,n,queueLoop.remove());
               m = addOrSubtractOne(forward, m);
               cont = checkCont(forward, m, cols - 1, i);
            }
            cont = checkCont(forward, n, cols - 1, i);
            while (cont) {
               set(m,n,queueLoop.remove());
               n = addOrSubtractOne(forward, n);
               cont = checkCont(forward, n, cols - 1, i);
            }
            forward = !forward;
         }

      }
   }
   
   // Only works on square matrices
   public void rotateEntriesCounter() {
      if (rows != cols) {
         throw new IllegalArgumentException();
      }
      
      for (int i = 0; i < rows / 2; i++) {  // i for each circle of the matrix
         int m = i;  // curr x pos
         int n = i;  // curr y pos
         boolean forward = true;
         boolean cont;
         Queue<E> queueLoop = new LinkedList();
         for (int j = 0; j < 2; j++) {
            cont = checkCont(forward, n, cols - 1, i);
            while (cont) {
               queueLoop.add(get(m,n));
               n = addOrSubtractOne(forward, n);
               cont = checkCont(forward, n, cols - 1, i);
            }
            cont = checkCont(forward, m, cols - 1, i);
            while (cont) {
               queueLoop.add(get(m,n));
               m = addOrSubtractOne(forward, m);
               cont = checkCont(forward, m, cols - 1, i);
            }
            forward = !forward;
         }
         for (int j = 0; j < rows - 1; j++) {
            queueLoop.add(queueLoop.remove());
         }
         for (int j = 0; j < 2; j++) {
            cont = checkCont(forward, n, cols - 1, i);
            while (cont) {
               set(m,n,queueLoop.remove());
               n = addOrSubtractOne(forward, n);
               cont = checkCont(forward, n, cols - 1, i);
            }
            cont = checkCont(forward, m, cols - 1, i);
            while (cont) {
               set(m,n,queueLoop.remove());
               m = addOrSubtractOne(forward, m);
               cont = checkCont(forward, m, cols - 1, i);
            }
            forward = !forward;
         }

      }
   }
   
   private boolean checkCont(boolean forward, int position, int limit, int inset) {
      if (forward) {
         return position < limit - inset;
      } else {
         return position > inset;
      }
   }
   
   private int addOrSubtractOne(boolean forward, int variable) {
      if (forward) {
         variable++;
      } else {
         variable--;
      }
      return variable;
   }
   
   
   // Only works on Numbers (excluding byte and short). Returns that number back in an array. Cast the
   //    returned array to whichever type you passed in. idk how to make this return a E[]
   public Number[] multiplyVector(E[] vector) {
      if (vector.length != cols || !(vector instanceof Number[])) {
         throw new IllegalArgumentException();
      }
            
      if (vector instanceof Integer[]) {
         Integer[] a = new Integer[rows];
            
         for (int i = 0; i < rows; i++) {
            int sum = 0;
            for (int j = 0; j < cols; j++) {
               int toAdd = (Integer) get(i, j) * (Integer) vector[j];
               sum += toAdd;
            }
            a[i] = sum;
         }
         return a;
      } else if (vector instanceof Double[]) {
         Double[] a = new Double[rows];
            
         for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols; j++) {
               double toAdd = (Double) get(i, j) * (Double) vector[j];
               sum += toAdd;
            }
            a[i] = sum;
         }
         return a;
      } else if (vector instanceof Float[]) {
         Float[] a = new Float[rows];
            
         for (int i = 0; i < rows; i++) {
            float sum = 0;
            for (int j = 0; j < cols; j++) {
               float toAdd = (Float) get(i, j) * (Float) vector[j];
               sum += toAdd;
            }
            a[i] = sum;
         }
         return a;
      } else if (vector instanceof Long[]) {
         Long[] a = new Long[rows];
            
         for (int i = 0; i < rows; i++) {
            long sum = 0;
            for (int j = 0; j < cols; j++) {
               long toAdd = (Long) get(i, j) * (Long) vector[j];
               sum += toAdd;
            }
            a[i] = sum;
         }
         return a;
      } else {
         throw new IllegalArgumentException();
      }
   }
   
   
   public Iterator<E> iterator() {
      return new cIt<E>(this);
   }
   
   private class cIt <E> implements Iterator<E> {
      private int i;
      private Matrix obj;
      
      private cIt(Matrix obj) {
         i = 0;
         this.obj = obj;
      }
      
      public boolean hasNext() {
         return i < obj.data.length;
      }
      
      public E next() {
         i++;
         return (E) obj.data[i - 1];
      }
   }
}