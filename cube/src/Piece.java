import java.util.Arrays;

public class Piece {
   private int[] orientation;
   private int id;
   private boolean oriMatters;
   
   public Piece(int id, int[] orientation, boolean oriMatters) {
      if (orientation.length != 3) {
         throw new IllegalArgumentException();
      }
      this.oriMatters = oriMatters;
      this.id = id;
      this.orientation = orientation.clone(); 
   }
   
   public int[] getOrientation() {
      return this.orientation.clone();
   }
   
   public int getID() {
      return id;
   }
   
   public String toString() {
      return "(" + Arrays.toString(orientation) + ", " + id + ", " + oriMatters + ")";
   }
   
   public void rotate(char axis, float angle) {
      if (oriMatters) {
         if ((axis != 'x' && axis != 'y' && axis != 'z') || (Math.abs(angle % 360) != 90 && Math.abs(angle % 360) != 270)) {
            throw new IllegalArgumentException();
         }
         
         orientation = integerToInt((Integer[]) Matrix.rotationalMatrix(axis, angle).multVec(intToInteger(orientation)));
      }
   }
   
   private Integer[] intToInteger(int[] a) {
      Integer[] output = new Integer[a.length];
      for (int i = 0; i < a.length; i++) {
         output[i] = a[i];
      }
      return output;
   }
   
   private int[] integerToInt(Integer[] a) {
      int[] output = new int[a.length];
      for (int i = 0; i < a.length; i++) {
         output[i] = a[i];
      }
      return output;
   }
   
   public boolean equals(Object other) {
	   if (this == other) {
		   return true;
	   }
	   if (other == null) {
		   return false;
	   }
	   if (this.getClass() != other.getClass()) {
		   return false;
	   }
	   Piece otherPiece = (Piece) other;
	   return id == otherPiece.id && oriMatters == otherPiece.oriMatters && (!oriMatters || orientation.equals(otherPiece.orientation));
   }
}