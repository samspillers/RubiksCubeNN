import java.util.*;

public class MoveID {
   private String id;
   public static final Set<MoveID> iDSet;
   private MoveID oppMove;
   
   private MoveID (String id) {
      this.id = id;
   }
   
   public String toString() {
      return id;
   }
   
   private void setOpp(MoveID opp) {
      oppMove = opp;
   }
   
   public boolean oppMove(MoveID in) {
      return oppMove == in;
   }
   
   public MoveID getOppMove() {
      return oppMove;
   }
   
   public static final MoveID F = new MoveID("F");
   public static final MoveID FP = new MoveID("F'");
   public static final MoveID B = new MoveID("B");
   public static final MoveID BP = new MoveID("B'");
   public static final MoveID U = new MoveID("U");
   public static final MoveID UP = new MoveID("U'");
   public static final MoveID D = new MoveID("D");
   public static final MoveID DP = new MoveID("D'");
   public static final MoveID L = new MoveID("L");
   public static final MoveID LP = new MoveID("L'");
   public static final MoveID R = new MoveID("R");
   public static final MoveID RP = new MoveID("R'");
   
   
   static {
      iDSet = new HashSet<MoveID>();
      iDSet.add(F);
      iDSet.add(FP);
      iDSet.add(B);
      iDSet.add(BP);
      iDSet.add(U);
      iDSet.add(UP);
      iDSet.add(D);
      iDSet.add(DP);
      iDSet.add(L);
      iDSet.add(LP);
      iDSet.add(R);
      iDSet.add(RP);
      
      F.setOpp(FP);
      FP.setOpp(F);
      B.setOpp(BP);
      BP.setOpp(B);
      U.setOpp(UP);
      UP.setOpp(U);
      D.setOpp(DP);
      DP.setOpp(D);
      L.setOpp(LP);
      LP.setOpp(L);
      R.setOpp(RP);
      RP.setOpp(R);
   }
}