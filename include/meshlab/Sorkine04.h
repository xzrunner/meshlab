#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace meshlab
{

class Sorkine04
{
public:
	Sorkine04(const std::vector<int>& cells, const float* positions, size_t positions_n, const std::vector<int>& roi, const int unconstrainedBegin, bool RSI);
	~Sorkine04();

	void doDeform(float* handlePositions, int nHandlePositions, float* outPositions);

private:
	void prepareDeform(const std::vector<int>& cells, const float* positions, size_t positions_n, const int unconstrainedBegin, bool RSI);

	std::vector<Eigen::Triplet<double>> calcUniformLaplacianCoeffs(std::vector<std::vector<int> > adj, std::vector<int>& rowBegins);

	std::vector<Eigen::Triplet<double>> calcEnergyMatrixCoeffs(
		const Eigen::VectorXd& roiPositions, const Eigen::VectorXd& delta, std::vector<std::vector<int>> adj,
		const std::vector<int>& rowBegins, const std::vector<Eigen::Triplet<double>>& laplacianCoeffs);

private:
	bool m_RSI;	// rotation scale invariant

	Eigen::VectorXd m_roiDelta;

	Eigen::SparseMatrix<double> m_augEnergyMatrixTrans;
	Eigen::SparseMatrix<double> m_augNormalizeDeltaCoordinatesTrans;

	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>* m_energyMatrixCholesky = nullptr;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>* m_normalizeDeltaCoordinatesCholesky = nullptr;

	std::vector<int> m_roi;

	Eigen::SparseMatrix<double> m_lapMat;

	std::vector<double> m_roiDeltaLengths;

	Eigen::VectorXd m_b;

}; // Sorkine04

}