// use code from: mikolalysenko/laplacian-deformation
// https://github.com/mikolalysenko/laplacian-deformation

#include "meshlab/Sorkine04.h"

namespace
{

double getLength(double ax, double ay, double az)
{
	return sqrt(ax*ax + ay * ay + az * az);
}

}

namespace ml
{

Sorkine04::Sorkine04(const std::vector<int>& cells, const float* positions, size_t positions_n, const std::vector<int>& roi, const int unconstrainedBegin, bool RSI)
	: m_roi(roi)
{
	prepareDeform(cells, positions, positions_n, unconstrainedBegin, RSI);
}

Sorkine04::~Sorkine04()
{
	if (m_energyMatrixCholesky != nullptr) {
		delete m_energyMatrixCholesky;
		m_energyMatrixCholesky = nullptr;
	}
	if (m_normalizeDeltaCoordinatesCholesky != nullptr) {
		delete m_normalizeDeltaCoordinatesCholesky;
		m_normalizeDeltaCoordinatesCholesky = nullptr;
	}
}

void Sorkine04::doDeform(float* newHandlePositions, int nHandlePositions, float* outPositions)
{
	{
		int count = 0;
		for (int i = 0; i < m_roiDelta.size(); ++i) {
			if (m_RSI) {
				// following from our derivations, we must set all these to zero.
				m_b[count++] = 0.0f;
			}
			else {
				m_b[count++] = m_roiDelta[i];
			}
		}
		for (int j = 0; j < nHandlePositions; ++j) {
			m_b[count++] = newHandlePositions[j * 3 + 0];
			m_b[count++] = newHandlePositions[j * 3 + 1];
			m_b[count++] = newHandlePositions[j * 3 + 2];
		}
	}

	Eigen::VectorXd minimizerSolution;
	{
		// Now we solve
		// Ax = b
		// where A is the energy matrix, and the value of b depends on whether we are optimizing (4) or (5)
		// by solving, we obtain the deformed surface coordinates that minimizes either (4) or (5).
		Eigen::VectorXd y = m_augEnergyMatrixTrans * m_b;
		minimizerSolution = m_energyMatrixCholesky->solve(y);
	}

	if (m_RSI) {
		// if minimizing (5), a local scaling is introduced by the solver.
		// so we need to normalize the delta coordinates of the deformed vertices back to their
		// original lengths.
		// otherwise, the mesh will increase in size when manipulating the mesh, which is not desirable.

		// the normalization step is pretty simple:
		// we find the delta coordinates of our solution.
		// then we normalize these delta coordinates, so that their lengths match the lengths of the original, undeformed delta coordinates.
		// then we simply do a minimization to find the coordinates that are as close as possible to the normalized delta coordinates
		// and the solution of this minimization is our final solution.

		Eigen::VectorXd solutionDelta = m_lapMat * minimizerSolution;

		int count = 0;
		for (int i = 0; i < m_roiDeltaLengths.size(); ++i) {

			double len = getLength(solutionDelta[3 * i + 0], solutionDelta[3 * i + 1], solutionDelta[3 * i + 2]);
			double originalLength = m_roiDeltaLengths[i];
			double scale = originalLength / len;

			for (int d = 0; d < 3; ++d) {
				m_b[count++] = scale * solutionDelta[3 * i + d];
			}
		}

		Eigen::VectorXd y = m_augNormalizeDeltaCoordinatesTrans * m_b;
		Eigen::VectorXd normalizedSolution = m_normalizeDeltaCoordinatesCholesky->solve(y);

		for (int i = 0; i < m_roi.size(); ++i) {
			for (int d = 0; d < 3; ++d) {
				outPositions[3 * m_roi[i] + d] = normalizedSolution[3 * i + d];
			}
		}
	}
	else {
		for (int i = 0; i < m_roi.size(); ++i) {
			for (int d = 0; d < 3; ++d) {
				outPositions[3 * m_roi[i] + d] = minimizerSolution[3 * i + d];
			}
		}
	}
}

void Sorkine04::prepareDeform(const std::vector<int>& cells, const float* positions, size_t positions_n, const int unconstrainedBegin, bool RSI)
{
	// free memory from previous call of prepareDeform()
	if (m_energyMatrixCholesky != nullptr) {
		delete m_energyMatrixCholesky;
		m_energyMatrixCholesky = nullptr;
	}
	if (m_normalizeDeltaCoordinatesCholesky != nullptr) {
		delete m_normalizeDeltaCoordinatesCholesky;
		m_normalizeDeltaCoordinatesCholesky = nullptr;
	}

	std::vector<std::vector<int> > adj;
	std::vector<int> roiMap(positions_n, -1);

	{
		for (int i = 0; i < m_roi.size(); ++i) {
			roiMap[m_roi[i]] = i;
		}

		adj.resize(m_roi.size());
		for (int i = 0; i < adj.size(); ++i) {
			adj[i] = std::vector<int>();
		}
		for (int i = 0; i < cells.size(); i += 3) {
			int c[3] = { cells[i + 0], cells[i + 1] , cells[i + 2] };

			for (int j = 0; j < 3; ++j) {
				int a = roiMap[c[j]];

				int b = roiMap[c[(j + 1) % 3]];

				if (a != -1 && b != -1) {
					adj[a].push_back(b);
				}
			}
		}
	}

	// put all the positions of the vertices in ROI in a single vector.
	Eigen::VectorXd roiPositions(m_roi.size() * 3);
	{
		int c = 0;
		for (int i = 0; i < m_roi.size(); ++i) {
			for (int d = 0; d < 3; ++d) {
				roiPositions[c++] = positions[3 * m_roi[i] + d];
			}
		}
	}

	std::vector<int> rowBegins;
	std::vector<Eigen::Triplet<double>> laplacianCoeffs;

	/*
	// cotangent laplacian doesnt yield any good results, for some reason :/
	so we don't use it. instead, use uniform.
	laplacianCoeffs = calcCotangentLaplacianCoeffs(

		cells, cells.size(),
		roiMap,

		m_roi.size(),
		adj,
		rowBegins,

		roiPositions);
		*/

	laplacianCoeffs = calcUniformLaplacianCoeffs(adj, rowBegins);

	m_lapMat = Eigen::SparseMatrix<double>(m_roi.size() * 3, m_roi.size() * 3);
	m_lapMat.setFromTriplets(laplacianCoeffs.begin(), laplacianCoeffs.end());

	// by simply multiplying by the laplacian matrix, we can compute the laplacian coordinates(the delta coordinates)
	// of the vertices in ROI.
	m_roiDelta = m_lapMat * roiPositions;

	// we save away the original lengths of the delta coordinates.
	// we need these when normalizing the results of our solver.
	{
		m_roiDeltaLengths = std::vector<double>(m_roiDelta.size() / 3, 0.0f);
		for (int i = 0; i < m_roiDelta.size() / 3; ++i) {
			m_roiDeltaLengths[i] = getLength(
				m_roiDelta[3 * i + 0],
				m_roiDelta[3 * i + 1],
				m_roiDelta[3 * i + 2]
			);
		}
	}

	std::vector<Eigen::Triplet<double>> energyMatrixCoeffs;

	// num rows in augmented matrix.
	// notice that we put x, y, and z in a large single matrix, and therefore it is multiplied by 3.
	int M = (m_roi.size() + unconstrainedBegin) * 3;
	// num columns in augmented matrix.
	int N = m_roi.size() * 3;

	if (RSI) {
		// this matrix represents the first term of the energy (5).
		energyMatrixCoeffs = calcEnergyMatrixCoeffs(roiPositions, m_roiDelta, adj, rowBegins, laplacianCoeffs);

		for (int i = 0; i < unconstrainedBegin; ++i) {
			laplacianCoeffs.push_back(Eigen::Triplet<double>(i * 3 + N + 0, 3 * i + 0, 1));
			laplacianCoeffs.push_back(Eigen::Triplet<double>(i * 3 + N + 1, 3 * i + 1, 1));
			laplacianCoeffs.push_back(Eigen::Triplet<double>(i * 3 + N + 2, 3 * i + 2, 1));
		}

		Eigen::SparseMatrix<double> augMat(M, N);
		augMat.setFromTriplets(laplacianCoeffs.begin(), laplacianCoeffs.end());
		m_augNormalizeDeltaCoordinatesTrans = augMat.transpose();

		m_normalizeDeltaCoordinatesCholesky = new Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>(m_augNormalizeDeltaCoordinatesTrans * augMat);
	}
	else {
		// if not rotation-scale-invariant, we simply use the regular laplacian matrix. This is the first term of the energy (4)
		energyMatrixCoeffs = laplacianCoeffs;
	}

	// in order to add the second term of the energy (4) or (5), we now augment the matrix.
	{
		// we augment the matrix by adding constraints for the handles.
		// these constraints ensure that if the handles are dragged, the handles will strictly follow in the specified direction.
		// the handle vertices are not free, unlike the unconstrained vertices.
		for (int i = 0; i < unconstrainedBegin; ++i) {
			energyMatrixCoeffs.push_back(Eigen::Triplet<double>(i * 3 + N + 0, 3 * i + 0, 1));
			energyMatrixCoeffs.push_back(Eigen::Triplet<double>(i * 3 + N + 1, 3 * i + 1, 1));
			energyMatrixCoeffs.push_back(Eigen::Triplet<double>(i * 3 + N + 2, 3 * i + 2, 1));
		}

		Eigen::SparseMatrix<double> augMat(M, N);
		augMat.setFromTriplets(energyMatrixCoeffs.begin(), energyMatrixCoeffs.end());
		m_augEnergyMatrixTrans = augMat.transpose();

		// for solving later, we need the cholesky decomposition of (transpose(augMat) * augMat)
		// this is a slow step! probably the slowest part of the entire algorithm.
		m_energyMatrixCholesky = new Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>(m_augEnergyMatrixTrans * augMat);
	}

	m_b = Eigen::VectorXd(M);
	m_RSI = RSI;
}

std::vector<Eigen::Triplet<double>> Sorkine04::calcUniformLaplacianCoeffs(std::vector<std::vector<int> > adj, std::vector<int>& rowBegins)
{
	std::vector<Eigen::Triplet<double>> result;
	std::map<int, double> row;

	for (int i = 0; i < (m_roi.size() * 3); ++i) {
		rowBegins.push_back(result.size());
		row.clear();

		row[(i % 3) + int(i / 3) * 3] = 1;
		double w = -1.0 / adj[int(i / 3)].size();
		for (int j = 0; j < adj[int(i / 3)].size(); ++j) {
			row[(i % 3) + 3 * adj[int(i / 3)][j]] = w;
		}

		for (const auto& p : row) {
			result.push_back(Eigen::Triplet<double>(i, p.first, p.second));
		}
	}
	rowBegins.push_back(result.size());

	return result;
}

std::vector<Eigen::Triplet<double>> Sorkine04::calcEnergyMatrixCoeffs(
	const Eigen::VectorXd& roiPositions, const Eigen::VectorXd& delta, std::vector<std::vector<int>> adj,
	const std::vector<int>& rowBegins, const std::vector<Eigen::Triplet<double>>& laplacianCoeffs)
{
	std::vector<Eigen::MatrixXd> Ts;

	Ts.resize(m_roi.size());

	for (int i = 0; i < m_roi.size(); ++i) {
		// set of {i} and the neigbbours of i.
		std::vector<int> iAndNeighbours;

		iAndNeighbours.push_back(i);
		for (int j = 0; j < adj[i].size(); ++j) {
			iAndNeighbours.push_back(adj[i][j]);
		}

		Eigen::MatrixXd At(7, iAndNeighbours.size() * 3);
		for (int row = 0; row < 7; ++row) {
			for (int col = 0; col < iAndNeighbours.size() * 3; ++col) {
				At(row, col) = 0.0f;
			}
		}

		for (int j = 0; j < iAndNeighbours.size(); ++j) {
			int k = iAndNeighbours[j];

			double vk[3];
			vk[0] = roiPositions[3 * k + 0];
			vk[1] = roiPositions[3 * k + 1];
			vk[2] = roiPositions[3 * k + 2];

			const int x = 0;
			const int y = 1;
			const int z = 2;

			At(0, j * 3 + 0) = +vk[x];
			At(1, j * 3 + 0) = 0;
			At(2, j * 3 + 0) = +vk[z];
			At(3, j * 3 + 0) = -vk[y];
			At(4, j * 3 + 0) = +1;
			At(5, j * 3 + 0) = 0;
			At(6, j * 3 + 0) = 0;

			At(0, j * 3 + 1) = +vk[y];
			At(1, j * 3 + 1) = -vk[z];
			At(2, j * 3 + 1) = 0;
			At(3, j * 3 + 1) = +vk[x];
			At(4, j * 3 + 1) = 0;
			At(5, j * 3 + 1) = +1;
			At(6, j * 3 + 1) = 0;

			At(0, j * 3 + 2) = +vk[z];
			At(1, j * 3 + 2) = +vk[y];
			At(2, j * 3 + 2) = -vk[x];
			At(3, j * 3 + 2) = 0;
			At(4, j * 3 + 2) = 0;
			At(5, j * 3 + 2) = 0;
			At(6, j * 3 + 2) = 1;
		}

		Eigen::MatrixXd invprod = (At * At.transpose()).inverse();
		Eigen::MatrixXd pseudoinv = invprod * At;
		Ts[i] = pseudoinv;
		// Ts[i] now contains (A^T A ) A^T (see equation 12 from paper.)
	}

	std::vector<Eigen::Triplet<double>> result;

	std::map<int, double> row;

	for (int i = 0; i < (m_roi.size() * 3); ++i) {
		row.clear();

		// add uniform weights to matrix(equation 2 from paper)
		for (int ientry = rowBegins[i]; ientry < rowBegins[i + 1]; ++ientry) {
			Eigen::Triplet<double> t = laplacianCoeffs[ientry];
			row[t.col()] = t.value();
		}

		// get delta coordinates for the vertex.
		double dx = delta[int(i / 3) * 3 + 0];
		double dy = delta[int(i / 3) * 3 + 1];
		double dz = delta[int(i / 3) * 3 + 2];

		std::vector<int> iAndNeighbours;
		iAndNeighbours.push_back(int(i / 3));
		for (int j = 0; j < adj[int(i / 3)].size(); ++j) {
			iAndNeighbours.push_back(adj[int(i / 3)][j]);
		}

		Eigen::MatrixXd T = Ts[int(i / 3)];

		Eigen::VectorXd s = T.row(0);
		Eigen::VectorXd h1 = T.row(1);
		Eigen::VectorXd h2 = T.row(2);
		Eigen::VectorXd h3 = T.row(3);
		Eigen::VectorXd tx = T.row(4);
		Eigen::VectorXd ty = T.row(5);
		Eigen::VectorXd tz = T.row(6);

		if ((i % 3) == 0) { // x case.
			for (int j = 0; j < T.row(0).size(); ++j) {
				int p = j % 3;
				int q = (int)floor((double)j / (double)3);
				int r = iAndNeighbours[q];

				row[p + 3 * r] -= dx * (+s[j]);
				row[p + 3 * r] -= dy * (-h3[j]);
				row[p + 3 * r] -= dz * (+h2[j]);
			}
		}
		else if ((i % 3) == 1) { // y case.
			for (int j = 0; j < T.row(0).size(); ++j) {
				int p = j % 3;
				int q = (int)floor((double)j / (double)3);
				int r = iAndNeighbours[q];

				row[p + 3 * r] -= dx * (+h3[j]);
				row[p + 3 * r] -= dy * (+s[j]);
				row[p + 3 * r] -= dz * (-h1[j]);
			}
		}
		else if ((i % 3) == 2) { // z case.
			for (int j = 0; j < T.row(0).size(); ++j) {
				int p = j % 3;
				int q = (int)floor((double)j / (double)3);
				int r = iAndNeighbours[q];

				row[p + 3 * r] -= dx * (-h2[j]);
				row[p + 3 * r] -= dy * (+h1[j]);
				row[p + 3 * r] -= dz * (+s[j]);

			}
		}

		for (const auto& p : row) {
			result.push_back(Eigen::Triplet<double>(i, p.first, p.second));
		}
	}

	return result;
}

}