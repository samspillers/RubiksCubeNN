import java.util.*;

public class RubiksCube {
	/*
	Rubik's cube matrix format:
	Matrix of arrays; think of matrix as the front face of the cube, and the arrays being the third
	dimension going back.
	Therefore all of the pieces in the first index of all of the arrays will be some kind of F piece.
	Matrix numbering begins in the top left corner, so 0,0 is front-up-left corner, and 3,3 is front-
	right-corner.
	Center-middle piece is null, so checks for that are present.
	The 'x' axis is the axis tangent to the F face, the 'y' axis is tangent to the U face, and the 'z'
	axis is tangent to the L face.
	*/
	
	private Matrix<Piece[]> cube;  // Holds all the cube's pieces
	private List<MoveID> movesList;  // Stores all moves performed on the cube from a solved state
	private final static int[] upOr = {1, 0, 0};  // Default orientation for a piece, points up
	private final static Matrix<Piece[]> solved3Cube = newCube(3);  // Holds a cube in a solved state for refence
	private int cubeDim;  // Holds the cube's size in one dimension. (Standard size is 3)
	
	// Creates cube in solved state
	public RubiksCube(int cubeDim) {
		this(cubeDim, null);
	}
	
	// Creates cube in state following given moves on a solved cube
	public RubiksCube(int cubeDim, List<MoveID> moves) {
		this.cubeDim = cubeDim;
		cube = newCube(cubeDim);
		movesList = new ArrayList<MoveID>();
		if (moves != null) {
			for (MoveID move : moves) {
				makeMove(move);
			}
		}
	}
	
	// x pos, y pos, z pos, starting at FUL corner. x and y are matrix pos, z is array pos (not client info, sorry)
	public Piece get(int i, int j, int k) {
		return cube.get(i, j)[k];
	}
	
	// Get's piece at index when counting all pieces from 0 to 25, skipping center piece
	//	Currently only works on cubeDim == 3 cube. TODO: Fix that
	public Piece get(int i) {
		if (i > 25) {
			throw new IllegalArgumentException();
		}
		if (i > 12) {
			i++;
		}
		return get(i % cubeDim, (i % (cubeDim * cubeDim)) / cubeDim, i / (cubeDim * cubeDim));
	}
	
	// Gets piece by its id, returns the piece. Returns null if doesn't exist
	// TODO: Is this method ever called? Why doesn't it return it's index in cube?
	public Piece getById(int id) {
		if (id >= 26) {
			throw new IllegalArgumentException();
		}
		
		for (int i = 0; i < 26; i++) {
			if (get(i).getID() == id) {
				return get(i);
			}
		}
		return null;
	}
	
	// Returns an array of all pieces in the order they appear on the cube, from index 0 to 25 
	public Piece[] getAllByPos() {
		Piece[] output = new Piece[cubeDim * cubeDim * cubeDim];
		for (int i = 0; i < cubeDim * cubeDim * cubeDim - 1; i++) {
			output[i] = get(i);
		}
		return output;
	}
	
	// Returns the positions of each piece in an array. Position of piece of id n is array[n], where
	//  	'array' is the returned array. Position ranges from 0 to 25 (higher if larger than dim 3)
	public int[] getAllByPiece() {
		int[] output = new int[cubeDim * cubeDim * cubeDim - 1];
		for (int i = 0; i < cubeDim * cubeDim * cubeDim - 1; i++) {
			output[get(i).getID()] = i;
		}
		return output;
	}
	
	// Return the moves performed on this cube to get the cube in the current state from a solved
	// 		state
	public List<MoveID> getMovesList() {
		return new ArrayList<MoveID>(movesList);
	}
	
	// Returns the list of moves that would undo the cube from its current state to the solved state
	// Returned move list has opposite moves in opposite order compared to getMovesList()
	public List<MoveID> getOppMovesList() {
		List<MoveID> output = new ArrayList<MoveID>();
		for (int i = movesList.size() - 1; i >= 0; i--) {
			output.add(movesList.get(i).getOppMove());
		}
		return output;
	}
	
	public boolean pieceCorrectPos(int x, int y, int z) {
		return cube.get(x, y)[z].getID() == solved3Cube.get(x, y)[z].getID();
	}
	
	public boolean pieceIsSolved(int x, int y, int z) {
		return cube.get(x, y)[z] == null || Arrays.equals(cube.get(x, y)[z].getOrientation(), upOr);
	}
	
	public boolean isSolved() {
		boolean output = true;
		for (Piece[] array : cube) {
			for (Piece piece : array) {
				if (piece != null) {
					output = output && Arrays.equals(piece.getOrientation(), upOr);
				}
			}
		}
		return output;
	}
	
	public void clearMovesList() {
		movesList.clear();
	}
	
	public void scrambleCube() {
		List<MoveID> possibleMoveList = new ArrayList<MoveID>(MoveID.iDSet);
		while (movesList.size() < 30) {
			int num = new Random().nextInt(possibleMoveList.size());
			makeMove(possibleMoveList.get(num));
		}
	}
	
	public void makeMove(MoveID move) {
		if (!MoveID.iDSet.contains(move)) {
			throw new IllegalArgumentException();
		}
		
		char axis;
		int row;
		boolean clockwise;
		
		if (move == MoveID.F || move == MoveID.FP || move == MoveID.B || move == MoveID.BP) {
			axis = 'x';
		} else if (move == MoveID.U || move == MoveID.UP || move == MoveID.D || move == MoveID.DP) {
			axis = 'y';
		} else {  // if (move == MoveID.L || move == MoveID.LP || move == MoveID.R || move == MoveID.RP)
			axis = 'z';
		}
		
		if (move == MoveID.F || move == MoveID.FP || move == MoveID.U || move == MoveID.UP || move == MoveID.L || move == MoveID.LP) {
			row = 0;
		} else {  // if (move == MoveID.B  || move == MoveID.BP || move == MoveID.D || move == MoveID.DP || move == MoveID.R || move == MoveID.RP)
			row = 2;
		}
		
		// Because of the weird way things are set up, UP is considered clockwise. Haven't looked into why this
		//    is the case enough, just know that this setup works.
		if (move == MoveID.FP || move == MoveID.UP || move == MoveID.L || move == MoveID.B || move == MoveID.D || move == MoveID.RP) {
			clockwise = true;
		} else {  // if (!above)
			clockwise = false;
		}
		
		Matrix<Piece> turningMatrix = getSpinMatrix(axis, row);
		
		if (clockwise) {
			turningMatrix.rotateEntriesClock();
		} else {
			turningMatrix.rotateEntriesCounter();
		}
		
		setSpinMatrix(axis, row, turningMatrix);
		
		if (movesList.size() != 0 && move.oppMove(movesList.get(movesList.size() - 1))) {
			movesList.remove(movesList.size() - 1);
		} else {
			movesList.add(move);
		}
		
		float angle;
		if (clockwise) {
			angle = 90;
		} else {
			angle = -90;
		}
		
		Matrix<Integer> rotate = Matrix.rotationalMatrix(axis, angle);
		
		for (Piece piece : turningMatrix) {
			piece.rotate(axis, angle);
		}
	}
	
	private Matrix<Piece> getSpinMatrix(char axis, int planeNum) {
		// int xSta, int xEnd, int ySta, int yEnd, int zSta, int zEnd) {
		if ((axis != 'x' && axis != 'y' && axis != 'z') || planeNum < 0 || planeNum > 2) {
			throw new IllegalArgumentException();
		}
		
		int xSta = 0;
		int xEnd = 3;
		int ySta = 0;
		int yEnd = 3;
		int zSta = 0;
		int zEnd = 3;
		
		if (axis == 'x') {
			xSta = planeNum;
			xEnd = planeNum + 1;
		} else if (axis == 'y') {
			ySta = planeNum;
			yEnd = planeNum + 1;
		} else {
			zSta = planeNum;
			zEnd = planeNum + 1;
		}
		
		Piece[] output = new Piece[9];
		int count = 0;
		
		for (int i = ySta; i < yEnd; i++) {
			for (int j = zSta; j < zEnd; j++) {
				for (int k = xSta; k < xEnd; k++) {
					if (count >= 9) {  // Shouldn't ever trigger
						throw new IllegalArgumentException();
					}
					output[count] = cube.get(i, j)[k];
					count++;
				}
			}
		}
		return new Matrix<Piece>(3, 3, output);
	}
	
	private void setSpinMatrix(char axis, int planeNum, Matrix<Piece> matrix) {
		if ((axis != 'x' && axis != 'y' && axis != 'z') || planeNum < 0 || planeNum > 2) {
			throw new IllegalArgumentException();
		}
		
		int xSta = 0;
		int xEnd = 3;
		int ySta = 0;
		int yEnd = 3;
		int zSta = 0;
		int zEnd = 3;
		
		if (axis == 'x') {
			xSta = planeNum;
			xEnd = planeNum + 1;
		} else if (axis == 'y') {
			ySta = planeNum;
			yEnd = planeNum + 1;
		} else {
			zSta = planeNum;
			zEnd = planeNum + 1;
		}
		
		Object[] input = matrix.getArray();
		int count = 0;
		
		for (int i = ySta; i < yEnd; i++) {
			for (int j = zSta; j < zEnd; j++) {
				for (int k = xSta; k < xEnd; k++) {
					if (count >= 9) {  // Shouldn't ever trigger
						throw new IllegalArgumentException();
					}
					cube.get(i,j)[k] = (Piece) input[count];
					count++;
				}
			}
		}
	}
	
	public String toString() {
		String output = "";
		for (int i = 0; i < 3; i++) {
			Matrix<Piece> matrix = new Matrix<Piece>(3, 3);
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					matrix.set(j, k, cube.get(j, k)[i]);
				}
			}
			output = output + matrix.toString();
			if (i < 2) {
				output = output + System.lineSeparator() + System.lineSeparator();
			}
		}
		return output;
	}
	
	// Bad cubeDim implementation. TODO: Fix to be expandable beyond size 3
	private static Matrix<Piece[]> newCube(int cubeDim) {
		Matrix<Piece[]> cube = new Matrix<Piece[]>(3, 3);
		for (int i = 0; i < cubeDim; i++) {
			for (int j = 0; j < cubeDim; j++) {
				cube.set(i, j, new Piece[3]);
			}
		}
		
		int l = 0;
		for (int i = 0; i < cubeDim; i++) {
			for (int j = 0; j < cubeDim; j++) {
				for (int k = 0; k < cubeDim; k++) {
					if ((i == 0 || i == cubeDim - 1) || (j == 0 || j == cubeDim - 1) || (k == 0 || k == cubeDim - 1)) {
						if (cubeDim % 2 == 1 && ((i == cubeDim / 2 && j == cubeDim / 2 && (k == 0 || k == cubeDim - 1)) || (i == cubeDim / 2 && k == cubeDim / 2 && (j == 0 || j == cubeDim - 1)) || (k == cubeDim / 2 && j == cubeDim / 2 && (i == 0 || i == cubeDim - 1)))) {
							cube.get(k, j)[i] = new Piece(l, upOr, false);
						} else {
							cube.get(k, j)[i] = new Piece(l, upOr, true);
						}
						l++;
					} else {
						cube.get(k, j)[i] = null;
					}
				}
			}
		}
		
		// The following 27 commands sets the correct pieces in the correct initial positions.
		return cube;
	}
}