#include <Eigen/Core>
#include <Eigen/Sparse>

#include <igl/triangle/triangulate.h>
#include <igl/intrinsic_delaunay_cotmatrix.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include "../include/MidedgeAngleTanFormulation.h"
#include "../include/RestState.h"
#include "../include/ElasticShell.h"
#include "../include/StVKMaterial.h"
#include "../include/MeshConnectivity.h"

#include <sstream>
#include <iomanip>
#include <fstream>

static double cotan(Eigen::Vector3d e0, Eigen::Vector3d e1)
{
    double dot = e0.dot(e1);
    double crossNorm = e0.cross(e1).norm();
    if (crossNorm == 0.0)
        return 0.0;
    return dot / crossNorm;
}

void makeRectangularMesh(double W, double L, int Lverts, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	int Wverts = std::max(2, int(W * Lverts / L));

    double triangleArea = 0.5 * W * L / (Lverts * Wverts);

    Eigen::MatrixXd Vin(2 * Wverts + 2 * Lverts, 2);
    Eigen::MatrixXi E(2 * Wverts + 2 * Lverts, 2);
    Eigen::MatrixXd dummyH(0, 2);
    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;

    int vrow = 0;
    int erow = 0;
    // top boundary
    for (int i = 1; i < Wverts; i++)
    {
        Vin(vrow, 0) = double(i) / double(Wverts) * W;
        Vin(vrow, 1) = L;
        if (i > 1)
        {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // bottom boundary
    for (int i = 1; i < Wverts; i++)
    {
        Vin(vrow, 0) = double(i) / double(Wverts) * W;
        Vin(vrow, 1) = 0;
        if (i > 1)
        {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // left boundary    
    for (int i = 0; i <= Lverts; i++)
    {
        Vin(vrow, 0) = 0;
        Vin(vrow, 1) = double(i) / double(Lverts) * L;
        if (i > 0)
        {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // right boundary    
    for (int i = 0; i <= Lverts; i++)
    {
        Vin(vrow, 0) = W;
        Vin(vrow, 1) = double(i) / double(Lverts) * L;
        if (i > 0)
        {
            E(erow, 0) = vrow - 1;
            E(erow, 1) = vrow;
            erow++;
        }
        vrow++;
    }
    // missing four edges
    E(erow, 0) = (Wverts - 1) - 1;
    E(erow, 1) = 2 * (Wverts - 1) + 2 * (Lverts + 1) - 1;
    erow++;
    E(erow, 0) = 2 * (Wverts - 1) + (Lverts + 1);
    E(erow, 1) = 2 * (Wverts - 1) - 1;
    erow++;
    E(erow, 0) = Wverts - 1;
    E(erow, 1) = 2 * (Wverts - 1);
    erow++;
    E(erow, 0) = 2 * (Wverts - 1) + (Lverts + 1) - 1;
    E(erow, 1) = 0;
    erow++;

    assert(vrow == 2 * Lverts + 2 * Wverts);
    assert(erow == 2 * Lverts + 2 * Wverts);
    std::stringstream ss;
    ss << "a" << std::setprecision(30) << std::fixed << triangleArea << "qDY";
    igl::triangle::triangulate(Vin, E, dummyH, ss.str(), V2, F2);

    // roll up

    int nverts = V2.rows();

    V.resize(nverts, 3);
    
    for (int i = 0; i < nverts; i++)
    {
        Eigen::Vector2d q = V2.row(i).transpose();
        V(i, 0) = q[0];
        V(i, 1) = q[1];
        V(i, 2) = 0;        
    }
    
    F = F2;
}

void findBoundary(const Eigen::MatrixXd& V, double tol, std::vector<int>& pinned)
{
	int nverts = V.rows();
    for (int i = 0; i < nverts; i++)
    {
        if (V(i, 1) < tol)
            pinned.push_back(i);
    }
}

void computeVertexAreas(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXd& vertexAreas)
{
    int nverts = V.rows();
    vertexAreas.resize(nverts);
    vertexAreas.setZero();

    int nfaces = F.rows();
    for (int i = 0; i < nfaces; i++)
    {
        Eigen::Vector3d v0 = V.row(F(i, 0)).transpose();
        Eigen::Vector3d v1 = V.row(F(i, 1)).transpose();
        Eigen::Vector3d v2 = V.row(F(i, 2)).transpose();
        double area = 0.5 * (v1 - v0).cross(v2 - v0).norm();
        for (int j = 0; j < 3; j++)
        {
            vertexAreas[F(i, j)] += area / 3.0;
        }
    }
}

void optimizeDOFs(bool useQuadraticBending,
    const Eigen::SparseMatrix<double> &Q,
    const LibShell::MeshConnectivity& mesh,
    Eigen::MatrixXd& curPos,
    Eigen::VectorXd& edgeDOFs,
    const LibShell::StVKMaterial<LibShell::MidedgeAngleTanFormulation>& mat,
    const LibShell::MonolayerRestState& restState,
    const std::vector<int> &pinned,
    const Eigen::VectorXd &extF,
    double tol)
{
    int nverts = curPos.rows();
    int nposdofs = nverts * 3;
    int nedgedofs = edgeDOFs.size();

    std::set<int> pinnedset;
    for (auto it : pinned)
        pinnedset.insert(it);

    std::vector<Eigen::Triplet<double> > Pcoeffs;

    int idx = 0;
    for (int i = 0; i < nverts; i++)
    {
        if (pinnedset.count(i))
            continue;
        for (int j = 0; j < 3; j++)
        {
            Pcoeffs.push_back({ idx, 3 * i + j, 1.0 });
            idx++;
        }
    }

    if (!useQuadraticBending)
    {
        for (int i = 0; i < nedgedofs; i++)
        {
            Pcoeffs.push_back({ idx, nposdofs + i, 1.0 });
            idx++;
        }
    }

    int reduceddofs = idx;

    std::vector<Eigen::Triplet<double> > Icoeffs;
    for (int i = 0; i < reduceddofs; i++)
    {        
        Icoeffs.push_back({ i, i, 1.0 });
    }
    Eigen::SparseMatrix<double> P(reduceddofs, nposdofs + nedgedofs);
    P.setFromTriplets(Pcoeffs.begin(), Pcoeffs.end());
    Eigen::SparseMatrix<double> I(reduceddofs, reduceddofs);
    I.setFromTriplets(Icoeffs.begin(), Icoeffs.end());

    Eigen::SparseMatrix<double> PT = P.transpose();


    double reg = 1e-6;
    while (true)
    {
        std::vector<Eigen::Triplet<double> > Hcoeffs;
        Eigen::VectorXd F;

        double origEnergy = 0;
        if (useQuadraticBending)
        {
            origEnergy = LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::elasticEnergy(mesh, curPos, edgeDOFs, mat, restState, LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::ET_STRETCHING, &F, &Hcoeffs, LibShell::HessianProjectType::kNone);
            Eigen::VectorXd q = curPos.reshaped<Eigen::RowMajor>();
            origEnergy += 0.5 * q.transpose() * Q * q;
            F.segment(0, nposdofs) += Q * q;

            for (int k = 0; k < Q.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Q, k); it; ++it)
                {
                    Hcoeffs.push_back({ (int)it.row(), (int)it.col(), it.value() });                    
                }
            }			
        }
        else
        {
            origEnergy = LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::elasticEnergy(mesh, curPos, edgeDOFs, mat, restState, &F, &Hcoeffs, LibShell::HessianProjectType::kNone);
        }

        origEnergy += -curPos.reshaped<Eigen::RowMajor>().dot(extF);

        F.segment(0, nposdofs) -= extF;
        Eigen::VectorXd PF = P * F;
        std::cout << "Force resid now: " << PF.norm() << ", energy: " << origEnergy << ", reg: " << reg;
        if (PF.norm() < tol)
        {
            std::cout << "; converged" << std::endl;
            return;
        }       

        Eigen::SparseMatrix<double> H(nposdofs + nedgedofs, nposdofs + nedgedofs);
        H.setFromTriplets(Hcoeffs.begin(), Hcoeffs.end());
        Hcoeffs.clear();
        
        Eigen::SparseMatrix<double> PHPT = P * H * PT;

        Eigen::SparseMatrix<double> M = PHPT + reg * I;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > solver(M);
        Eigen::VectorXd update = solver.solve(-PF);
        if (solver.info() != Eigen::Success) {
            std::cout << "; solve failed" << std::endl;
            reg *= 2.0;
            continue;
        }
        else
        {
            std::cout << ", Newton decrement: " << update.norm() << std::endl;
        }

        if (update.norm() < 1e-12)
        {
            std::cout << "Stalled" << std::endl;
            return;
        }

        Eigen::VectorXd fullUpdate = PT * update;
        Eigen::MatrixXd newPos = curPos;
        for (int i = 0; i < nverts; i++)
        {
            newPos.row(i) += fullUpdate.segment<3>(3 * i).transpose();
        }

        Eigen::VectorXd newedgeDOFs = edgeDOFs + fullUpdate.segment(nposdofs, nedgedofs);

        if (!useQuadraticBending)
        {
            if (!LibShell::MidedgeAngleTanFormulation::edgeDOFsValid(mesh, newPos, newedgeDOFs))
            {
                std::cout << "Edge DOFs invalid" << std::endl;
                reg *= 2.0;
                continue;
            }
        }

        double newenergy = 0;
        if (useQuadraticBending)
        {
            newenergy = LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::elasticEnergy(mesh, newPos, newedgeDOFs, mat, restState, LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::ET_STRETCHING, NULL, NULL);
            Eigen::VectorXd q = newPos.reshaped<Eigen::RowMajor>();
            newenergy += 0.5 * q.transpose() * Q * q;
        }
        else
        {
            newenergy = LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::elasticEnergy(mesh, newPos, newedgeDOFs, mat, restState, NULL, NULL);
        }
            
        newenergy += -newPos.reshaped<Eigen::RowMajor>().dot(extF);

        if (newenergy > origEnergy)
        {
            std::cout << "Not a descent step, " << origEnergy << " -> " << newenergy << std::endl;
            reg *= 2.0;
            continue;
        }
        curPos = newPos;
        edgeDOFs = newedgeDOFs;
        reg *= 0.5;
    }
}

void buildQBCRBendingMatrix(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    double thickness,
    double lameAlpha,
    double lameBeta,
    Eigen::SparseMatrix<double>& bendingM)
{
    int nverts = restPos.rows();
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();

    double weight = thickness * thickness * thickness / 12.0 * (lameAlpha + 2.0 * lameBeta);
    std::vector<Eigen::Triplet<double>> Qentries;
    for (int i = 0; i < nedges; i++)
    {
        if (mesh.edgeOppositeVertex(i, 0) == -1 || mesh.edgeOppositeVertex(i, 1) == -1)
            continue; // skip boundary edges    
        int v[4];
        v[0] = mesh.edgeVertex(i, 0);
        v[1] = mesh.edgeVertex(i, 1);
        v[2] = mesh.edgeOppositeVertex(i, 0);
        v[3] = mesh.edgeOppositeVertex(i, 1);
        Eigen::Vector3d e0 = restPos.row(v[1]) - restPos.row(v[0]);
        Eigen::Vector3d e1 = restPos.row(v[2]) - restPos.row(v[0]);
        Eigen::Vector3d e2 = restPos.row(v[3]) - restPos.row(v[0]);
        Eigen::Vector3d e3 = restPos.row(v[2]) - restPos.row(v[1]);
        Eigen::Vector3d e4 = restPos.row(v[3]) - restPos.row(v[1]);
        double c01 = cotan(e0, e1);
        double c02 = cotan(e0, e2);
        double c03 = cotan(-e0, e3);
        double c04 = cotan(-e0, e4);
        Eigen::Vector4d K0(
            c03 + c04,
            c01 + c02,
            -c01 - c03,
            -c02 - c04);
        double A0 = e0.cross(e1).norm() / 2.0;
        double A1 = e0.cross(e2).norm() / 2.0;
        Eigen::Matrix4d Q = 3.0 / (A0 + A1) * weight * K0 * K0.transpose();
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    if (Q(j, k) != 0.0)
                    {
                        Qentries.emplace_back(3 * v[j] + l, 3 * v[k] + l, Q(j, k));
                    }
                }
            }
        }
    }
    bendingM.resize(3 * nverts, 3 * nverts);
    bendingM.setFromTriplets(Qentries.begin(), Qentries.end());
}

void buildQBVBendingMatrix(
    const LibShell::MeshConnectivity& mesh,
    const Eigen::MatrixXd& restPos,
    double thickness,
    double lameAlpha,
    double lameBeta,
    Eigen::SparseMatrix<double>& bendingM)
{
    int nverts = restPos.rows();
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();

    Eigen::SparseMatrix<double> L;
    igl::intrinsic_delaunay_cotmatrix(restPos, mesh.faces(), L);
    std::vector<Eigen::Triplet<double> > bigLcoeffs;
    for (int k = 0; k < L.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
        {
            for (int j = 0; j < 3; j++)
            {
                bigLcoeffs.push_back({ 3 * (int)it.row() + j, 3 * (int)it.col() + j, it.value() });
            }
        }
    }

    Eigen::SparseMatrix<double> bigL(3 * nverts, 3 * nverts);
    bigL.setFromTriplets(bigLcoeffs.begin(), bigLcoeffs.end());

    std::vector<bool> bdry(nverts);
    for (int i = 0; i < nedges; i++)
    {
        if (mesh.edgeFace(i, 0) == -1 || mesh.edgeFace(i, 1) == -1)
        {
            int v0idx = mesh.edgeVertex(i, 0);
            int v1idx = mesh.edgeVertex(i, 1);
            bdry[v0idx] = true;
            bdry[v1idx] = true;
        }
    }

    Eigen::MatrixXd edgeLengths;
    igl::edge_lengths(restPos, mesh.faces(), edgeLengths);

    Eigen::MatrixXi newFaces;
    Eigen::MatrixXd newLengths;
    igl::intrinsic_delaunay_triangulation(edgeLengths, mesh.faces(), newLengths, newFaces);
    int nnewfaces = newFaces.rows();

    std::vector<Eigen::Triplet<double> > Ncoeffs;
    for (int i = 0; i < nnewfaces; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int v0idx = newFaces(i, j);
            int v1idx = newFaces(i, (j + 1) % 3);
            int v2idx = newFaces(i, (j + 2) % 3);
            if (bdry[v0idx] && bdry[v1idx])
            {
                double l0 = newLengths(i, j);
                double l1 = newLengths(i, (j + 1) % 3);
                double l2 = newLengths(i, (j + 2) % 3);
                double S = 0.25 * sqrt((l0 + l1 + l2) * (l0 + l1 - l2) * (l0 - l1 + l2) * (-l0 + l1 + l2));
                double cot0 = (l1 * l1 + l2 * l2 - l0 * l0) / (4.0 * S);
                double cot1 = (l0 * l0 + l2 * l2 - l1 * l1) / (4.0 * S);
                for (int k = 0; k < 3; k++)
                {
                    Ncoeffs.push_back({ 3 * v0idx + k, 3 * v0idx + k, 0.5 * cot1 });
                    Ncoeffs.push_back({ 3 * v0idx + k, 3 * v1idx + k, 0.5 * cot0 });
                    Ncoeffs.push_back({ 3 * v0idx + k, 3 * v2idx + k, 0.5 * (-cot0 - cot1) });
                    Ncoeffs.push_back({ 3 * v1idx + k, 3 * v0idx + k, 0.5 * cot1 });
                    Ncoeffs.push_back({ 3 * v1idx + k, 3 * v1idx + k, 0.5 * cot0 });
                    Ncoeffs.push_back({ 3 * v1idx + k, 3 * v2idx + k, 0.5 * (-cot0 - cot1) });
                }
            }
        }
    }

    Eigen::SparseMatrix<double> N(3 * nverts, 3 * nverts);
    N.setFromTriplets(Ncoeffs.begin(), Ncoeffs.end());

    std::vector<double> Mcoeffs(nverts);
    std::vector<double> energycoeffs(nverts);
    std::vector<Eigen::Triplet<double> > Minvcoeffs;

    for (int i = 0; i < nfaces; i++)
    {
        double weight = thickness * thickness * thickness / 12.0 * (lameAlpha + 2.0 * lameBeta);
        Eigen::Vector3d e1 = restPos.row(mesh.faceVertex(i, 1)) - restPos.row(mesh.faceVertex(i, 0));
        Eigen::Vector3d e2 = restPos.row(mesh.faceVertex(i, 2)) - restPos.row(mesh.faceVertex(i, 0));
        double area = 0.5 * e1.cross(e2).norm();

        for (int j = 0; j < 3; j++)
        {
            int vidx = mesh.faceVertex(i, j);
            Mcoeffs[vidx] += area / 3.0;
            energycoeffs[vidx] += weight * area / 3.0;
        }
    }
    for (int i = 0; i < nverts; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Minvcoeffs.push_back({ 3 * i + j, 3 * i + j, energycoeffs[i] / Mcoeffs[i] / Mcoeffs[i] });
        }
    }
    Eigen::SparseMatrix<double> Minv(3 * nverts, 3 * nverts);
    Minv.setFromTriplets(Minvcoeffs.begin(), Minvcoeffs.end());

    bendingM = (bigL.transpose() + N.transpose()) * Minv * (bigL + N);
}

int main()
{
    std::string logfile = "log.txt";
    std::ofstream log(logfile);

//    polyscope::init();
    
    double nu = 0.35;
	double L = 1.0;
	double D = 1.0;
	double h = 1e-3;

    double density = 1; // doesn't matter, will divide out

	// D = young * thickness^3 / (12 * (1 - poisson^2))

	double Y = D * 12 * (1 - nu * nu) / (h * h * h);

    double lameAlpha = Y * nu / (1.0 - nu * nu);
    double lameBeta = Y / 2.0 / (1.0 + nu);

    double newtonTol = 1e-7;


	double initialW = 0.1;
	double dW = 0.1;
	double Wsteps = 9;
	for (int i = 0; i < Wsteps; i++)
	{
		double W = initialW + i * dW;

        log << "Testing W = " << W << std::endl;

		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		makeRectangularMesh(W, L, 100, V, F);
        int nverts = V.rows();

        std::vector<int> pinned;
        findBoundary(V, 1.5 * L/100.0, pinned);

        Eigen::VectorXd vertexAreas;
        computeVertexAreas(V, F, vertexAreas);
        for (auto it : pinned)
            vertexAreas[it] = 0;

        LibShell::MeshConnectivity mesh(F);

        Eigen::VectorXd edgeDOFs;
        LibShell::MidedgeAngleTanFormulation::initializeExtraDOFs(edgeDOFs, mesh, V);

        // initialize the rest geometry of the shell
        LibShell::MonolayerRestState restState;

        // set uniform thicknesses
        restState.thicknesses.resize(mesh.nFaces(), h);
        restState.lameAlpha.resize(mesh.nFaces(), lameAlpha);
        restState.lameBeta.resize(mesh.nFaces(), lameBeta);

        LibShell::StVKMaterial<LibShell::MidedgeAngleTanFormulation> mat;

        // initialize first and second fundamental forms to those of input mesh
        LibShell::ElasticShell<LibShell::MidedgeAngleTanFormulation>::
            firstFundamentalForms(mesh, V, restState.abars);

        restState.bbars.resize(mesh.nFaces());
        for (int i = 0; i < mesh.nFaces(); i++)
        {
            restState.bbars[i].setZero();
        }        

        /*std::stringstream ss;
        ss << "mesh-" << W;
        auto surf = polyscope::registerSurfaceMesh(ss.str(), V, F);*/


        Eigen::MatrixXd S1V = V;
        Eigen::VectorXd S1edgeDOFs = edgeDOFs;

        Eigen::VectorXd perturbation(nverts);
        perturbation.setRandom();
        for (auto it : pinned)
            perturbation[it] = 0;
        for (int i = 0; i < nverts; i++)
        {
            S1V(i, 2) += 1e-4 * std::fabs(perturbation[i]);
        }
        

        Eigen::MatrixXd QBVV = S1V;
        Eigen::VectorXd QBVedgeDOFs = edgeDOFs;
        Eigen::SparseMatrix<double> QV;
        buildQBVBendingMatrix(mesh, V, h, lameAlpha, lameBeta, QV);

        Eigen::MatrixXd QBCRV = S1V;
        Eigen::VectorXd QBCRedgeDOFs = edgeDOFs;
        Eigen::SparseMatrix<double> QCR;
        buildQBCRBendingMatrix(mesh, V, h, lameAlpha, lameBeta, QCR);

        for (double gammaStar = 40; gammaStar >= 10; gammaStar -= 0.5)
        {
            double g = D * gammaStar / (density * L * L * L * h);

            Eigen::VectorXd extF(3 * nverts);
            extF.setZero();
            for (int i = 0; i < nverts; i++)
            {
                extF(3 * i + 0) = g * h * vertexAreas[i] * density;
            }

            std::cout << "==========" << std::endl;
            std::cout << "Starting solve: w/L = " << W << ", gammaStar = " << gammaStar << std::endl;
            std::cout << "==========" << std::endl;
            optimizeDOFs(false, QV, mesh, S1V, S1edgeDOFs, mat, restState, pinned, extF, newtonTol);
            optimizeDOFs(true, QV, mesh, QBVV, QBVedgeDOFs, mat, restState, pinned, extF, newtonTol);
            optimizeDOFs(true, QCR, mesh, QBCRV, QBCRedgeDOFs, mat, restState, pinned, extF, newtonTol);

            double maxS1Lateral = 0.0;
            double maxQBVLateral = 0.0;
            double maxQBCRLateral = 0.0;
            for (int i = 0; i < nverts; i++)
            {
                maxS1Lateral = std::max(maxS1Lateral, std::fabs(S1V(i, 2)));
                maxQBVLateral = std::max(maxQBVLateral, std::fabs(QBVV(i, 2)));
                maxQBCRLateral = std::max(maxQBCRLateral, std::fabs(QBCRV(i, 2)));
            }

            log << gammaStar << ": " << maxS1Lateral << ", " << maxQBVLateral << ", " << maxQBCRLateral << std::endl;
            
            /*{
                std::stringstream ss;
                ss << "mesh-S1-" << W << "-" << gammaStar;
                auto surf = polyscope::registerSurfaceMesh(ss.str(), S1V, F);
                surf->setEnabled(false);
            }
            {
                std::stringstream ss;
                ss << "mesh-QBV-" << W << "-" << gammaStar;
                auto surf = polyscope::registerSurfaceMesh(ss.str(), QBVV, F);
                surf->setEnabled(false);
            }
            {
                std::stringstream ss;
                ss << "mesh-QBCR-" << W << "-" << gammaStar;
                auto surf = polyscope::registerSurfaceMesh(ss.str(), QBCRV, F);
                surf->setEnabled(false);
            }*/
        }        
	}

    //polyscope::show();
}