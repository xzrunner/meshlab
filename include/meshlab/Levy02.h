/*
 *  Copyright (c) 2004-2010, Bruno Levy
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *  * Neither the name of the ALICE Project-Team nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  If you modify this software, you should include a notice giving the
 *  name of the person performing the modification, the date of modification,
 *  and the reason for such modification.
 *
 *  Contact: Bruno Levy
 *
 *     levy@loria.fr
 *
 *     ALICE Project
 *     LORIA, INRIA Lorraine,
 *     Campus Scientifique, BP 239
 *     54506 VANDOEUVRE LES NANCY CEDEX
 *     FRANCE
 *
 */

#pragma once

#include "meshlab/Mesh.h"

namespace meshlab
{

class Levy02
{
public:
    Levy02(const std::vector<vec3d>& verts, const std::vector<size_t>& tris);

    /**
     * \brief Computes the least squares conformal map and stores it in
     *  the texture coordinates of the mesh.
     * \details Outline of the algorithm (steps 1,2,3 are not used
     *   in spetral mode):
     *   - 1) Find an initial solution by projecting on a plane
     *   - 2) Lock two vertices of the mesh
     *   - 3) Copy the initial u,v coordinates to OpenNL
     *   - 4) Construct the LSCM equation with OpenNL
     *   - 5) Solve the equation with OpenNL
     *   - 6) Copy OpenNL solution to the u,v coordinates
     */
    void Apply();

    auto& GetUVs() const { return m_uvs; }

private:
    /**
     * \brief Chooses an initial solution, and locks two vertices.
     */
    void Project();

    /**
     * \brief Computes the coordinates of the vertices of a triangle
     * in a local 2D orthonormal basis of the triangle's plane.
     * \param[in] p0 , p1 , p2 the 3D coordinates of the vertices of
     *   the triangle
     * \param[out] z0 , z1 , z2 the 2D coordinates of the vertices of
     *   the triangle
     */
    void ProjectTriangle(const vec3d& p0, const vec3d& p1, const vec3d& p2,
        vec2d& z0, vec2d& z1, vec2d& z2);

    /**
     * \brief Copies u,v coordinates from the mesh to OpenNL solver.
     */
    void MeshToSolver();
    /**
     * \brief Copies u,v coordinates from OpenNL solver to the mesh.
     */
    void SolverToMesh();

    /**
     * \brief Creates the LSCM equations in OpenNL.
     */
    void SetupLSCM();
    /**
     * \brief Creates the LSCM equation in OpenNL, related with
     *   a given triangle, specified by vertex indices.
     * \param[in] v0 , v1 , v2 the indices of the three vertices of
     *   the triangle.
     * \details Uses the geometric form of LSCM equation:
     *  (Z1 - Z0)(U2 - U0) = (Z2 - Z0)(U1 - U0)
     *  Where Uk = uk + i.vk is the complex number
     *                       corresponding to (u,v) coords
     *       Zk = xk + i.yk is the complex number
     *                       corresponding to local (x,y) coords
     * There is no divide with this expression,
     *  this makes it more numerically stable in
     * the presence of degenerate triangles.
     */
    void SetupConformalMapRelations(size_t v0, size_t v1, size_t v2);

    /**
     * \brief Translates and scales tex coords in such a way that they fit
     * within the unit square.
     */
    void NormalizeUV();

private:
    const std::vector<vec3d>&  m_verts;
    const std::vector<size_t>& m_tris;

    std::vector<vec2d> m_uvs;

    int m_vxmin = -1;
    int m_vxmax = -1;

}; // Levy02

}