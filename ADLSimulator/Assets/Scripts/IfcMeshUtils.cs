using GeometryGym.Ifc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

class IfcMeshUtils
{
    public enum FaceSide { FRONT, BACK, BOTH }

    public static Vector2[] CalculateUVs(Mesh mesh, List<Vector3> newVerticesFinal)
    {
        // calculate UVs ============================================
        float scaleFactor = 0.5f;
        Vector2[] uvs = new Vector2[newVerticesFinal.Count];
        int len = mesh.GetIndices(0).Length;
        int[] idxs = mesh.GetIndices(0);
        for (int i = 0; i < len; i = i + 3)
        {
            Vector3 v1 = newVerticesFinal[idxs[i + 0]];
            Vector3 v2 = newVerticesFinal[idxs[i + 1]];
            Vector3 v3 = newVerticesFinal[idxs[i + 2]];
            Vector3 normal = Vector3.Cross(v3 - v1, v2 - v1);
            Quaternion rotation;
            if (normal == Vector3.zero)
                rotation = new Quaternion();
            else
                rotation = Quaternion.Inverse(Quaternion.LookRotation(normal));
            uvs[idxs[i + 0]] = (Vector2)(rotation * v1) * scaleFactor;
            uvs[idxs[i + 1]] = (Vector2)(rotation * v2) * scaleFactor;
            uvs[idxs[i + 2]] = (Vector2)(rotation * v3) * scaleFactor;
        }
        //==========================================================
        return uvs;
    }

    public static List<Vector3> PositionElementAndGetBB(IfcElement elem, List<Vector3> newVertices, out BoundingBox boundingBox, float scale)
    {
        if (newVertices.Count == 0)
        {
            boundingBox = null;
            return null;
        }

        float minVx = newVertices.Min(v => v.x);
        float maxVx = newVertices.Max(v => v.x);
        float minVy = newVertices.Min(v => v.y);
        float maxVy = newVertices.Max(v => v.y);
        float minVz = newVertices.Min(v => v.z);
        float maxVz = newVertices.Max(v => v.z);
        Vector4 centerV = new Vector4((maxVx + minVx) / 2, (maxVy + minVy) / 2, (maxVz + minVz) / 2, 1);
        Vector3 dimsV = new Vector3((maxVx - minVx), (maxVy - minVy), (maxVz - minVz));
        Vector4 centerVright = new Vector4(centerV.x + 1.0f, centerV.y, centerV.z, 1);
        Vector4 centerVforward = new Vector4(centerV.x, centerV.y + 1.0f, centerV.z, 1);
        Vector4 centerVup = new Vector4(centerV.x, centerV.y, centerV.z + 1.0f, 1);

        // Need to position the elements in the correct location:
        IfcLocalPlacement localPlace = elem.Placement as IfcLocalPlacement;
        IfcAxis2Placement3D axisPlace = localPlace.RelativePlacement as IfcAxis2Placement3D;
        Matrix4x4 matrixTrans = ExtracMatrixAxis2Placement3D(axisPlace);

        IfcLocalPlacement relLocalPlace = localPlace.PlacementRelTo as IfcLocalPlacement;
        while (relLocalPlace != null)
        {
            localPlace = relLocalPlace;
            axisPlace = localPlace.RelativePlacement as IfcAxis2Placement3D;
            Matrix4x4 matrixTransTemp = ExtracMatrixAxis2Placement3D(axisPlace);
            matrixTrans = matrixTransTemp * matrixTrans;
            relLocalPlace = localPlace.PlacementRelTo as IfcLocalPlacement;
        }

        Vector4 newColumn = new Vector4(matrixTrans.GetColumn(3).x * scale, matrixTrans.GetColumn(3).y * scale, matrixTrans.GetColumn(3).z * scale, matrixTrans.GetColumn(3).w);
        matrixTrans.SetColumn(3, newColumn);

        List <Vector3> newVerticesFinal = new List<Vector3>();
        foreach (Vector3 vect in newVertices)
        {
            Vector4 vect4 = new Vector4(vect.x, vect.y, vect.z, 1);
            vect4 = matrixTrans * vect4;
            Vector3 newVect = new Vector3(vect4.x, vect4.z, vect4.y);
            newVerticesFinal.Add(newVect);
        }

        // For the Bounding Box:
        Vector3 newCenterV = matrixTrans * centerV;
        newCenterV = new Vector3(newCenterV.x, newCenterV.z, newCenterV.y);
        Vector3 newCenterVforward = matrixTrans * centerVforward;
        newCenterVforward = new Vector3(newCenterVforward.x, newCenterVforward.z, newCenterVforward.y);
        Vector3 newCenterVup = matrixTrans * centerVup;
        newCenterVup = new Vector3(newCenterVup.x, newCenterVup.z, newCenterVup.y);
        Vector3 newCenterVright = matrixTrans * centerVright;
        newCenterVright = new Vector3(newCenterVright.x, newCenterVright.z, newCenterVright.y);
        Vector3 newDimsV = new Vector3(dimsV.x, dimsV.z, dimsV.y);
        boundingBox = new BoundingBox(newCenterV, dimsV, newCenterVforward - newCenterV, newCenterVup - newCenterV, newCenterVright - newCenterV);

        return newVerticesFinal;
    }

    public static void GetTrianglesFromShapeRepresentation(ref List<Vector3> newVertices, ref List<int> newTriangles, IfcShapeRepresentation shapeRepModel)
    {
        //Debug.Log(shapeRepModel.RepresentationType);

        // Combo method:
        if (shapeRepModel.RepresentationType == "MappedRepresentation")
        {
            foreach (IfcRepresentationItem repItem in shapeRepModel.Items)
            {
                IfcMappedItem item = repItem as IfcMappedItem;
                IfcShapeRepresentation sourceShapeRep = item.MappingSource.MappedRepresentation as IfcShapeRepresentation;
                GetTrianglesFromShapeRepresentation(ref newVertices, ref newTriangles, sourceShapeRep);
            }
            return;
        }

        //// Face based methods:
        if (shapeRepModel.RepresentationType == "SurfaceModel")
        {
            foreach (IfcRepresentationItem repItem in shapeRepModel.Items)
            {
                IfcShellBasedSurfaceModel item = repItem as IfcShellBasedSurfaceModel;
                if (item != null)
                {
                    List<IfcShell> shells = item.SbsmBoundary.ToList();
                    foreach (IfcShell shell in shells)
                    {
                        List<IfcFace> faces = shell.CfsFaces.ToList();
                        CreateTrianglesFromFaceList(ref newVertices, ref newTriangles, faces, FaceSide.FRONT);
                    }
                }

                IfcFaceBasedSurfaceModel item2 = repItem as IfcFaceBasedSurfaceModel;
                if (item2 != null)
                {
                    foreach (IfcConnectedFaceSet conFaceSet in item2.FbsmFaces)
                    {
                        List<IfcFace> faces = conFaceSet.CfsFaces.ToList();
                        CreateTrianglesFromFaceList(ref newVertices, ref newTriangles, faces, FaceSide.FRONT);
                    }
                }
            }
            return;
        }
        if (shapeRepModel.RepresentationType == "Brep")
        {
            foreach (IfcRepresentationItem repItem in shapeRepModel.Items)
            {
                IfcFacetedBrep item = repItem as IfcFacetedBrep;
                List<IfcFace> faces = item.Outer.CfsFaces.ToList();
                CreateTrianglesFromFaceList(ref newVertices, ref newTriangles, faces, FaceSide.FRONT);
            }
            return;
        }
        if (shapeRepModel.RepresentationType == "GeometricSet")
        {
            foreach (IfcRepresentationItem repItem in shapeRepModel.Items)
            {
                IfcGeometricSet item = repItem as IfcGeometricSet;
                List<IfcFace> faces = item.Extract<IfcFace>().ToList();
                CreateTrianglesFromFaceList(ref newVertices, ref newTriangles, faces, FaceSide.FRONT);
            }
            return;
        }

        //// Line based methods:
        if (shapeRepModel.RepresentationType == "SweptSolid")
        {
            foreach (IfcRepresentationItem repItem in shapeRepModel.Items)
            {
                IfcSweptAreaSolid solidModel = repItem as IfcSweptAreaSolid;
                IfcExtrudedAreaSolid extrudedAreaSolid = solidModel as IfcExtrudedAreaSolid;
                List<Vector3> allPoints;
                List<int> triangleNums;
                GetTrianglesForExtrudedSolid(extrudedAreaSolid, out allPoints, out triangleNums);

                int verticiesCount = newVertices.Count;
                List<int> triangleNumsFixed = new List<int>();
                foreach (int tN in triangleNums)
                {
                    triangleNumsFixed.Add(tN + verticiesCount);
                }
                newTriangles.AddRange(triangleNumsFixed);
                newVertices.AddRange(allPoints);
            }

            return;
        }
        if (shapeRepModel.RepresentationType == "Clipping")
        {
            foreach (IfcRepresentationItem repItem in shapeRepModel.Items)
            {
                IfcBooleanClippingResult item = repItem as IfcBooleanClippingResult;
                List<Vector3> allPoints;
                List<int> triangleNums;
                GetTriangleFromClippedItem(item, out allPoints, out triangleNums);

                int verticiesCount = newVertices.Count;
                List<int> triangleNumsFixed = new List<int>();
                foreach (int tN in triangleNums)
                {
                    triangleNumsFixed.Add(tN + verticiesCount);
                }
                newTriangles.AddRange(triangleNumsFixed);
                newVertices.AddRange(allPoints);
            }

            return;
        }
    }

    private static void GetTrianglesForExtrudedSolid(IfcExtrudedAreaSolid extrudedAreaSolid, out List<Vector3> allPoints, out List<int> triangleNums)
    {
        double depth = extrudedAreaSolid.Depth;
        double xDir = extrudedAreaSolid.ExtrudedDirection.DirectionRatioX;
        double yDir = extrudedAreaSolid.ExtrudedDirection.DirectionRatioY;
        double zDir = extrudedAreaSolid.ExtrudedDirection.DirectionRatioZ;
        IfcProfileDef profDef = extrudedAreaSolid.SweptArea;

        // Need to position the elements in the correct location:
        IfcAxis2Placement3D axisPlace = extrudedAreaSolid.Position;
        Matrix4x4 matrixTrans = ExtracMatrixAxis2Placement3D(axisPlace);
        List<Vector3> polyLinePoints = GetExtrudeFacePoints(profDef);

        List<Vector3> allPointsTemp = new List<Vector3>();
        foreach (Vector3 point in polyLinePoints)
        {
            allPointsTemp.Add(point);
        }
        foreach (Vector3 point in polyLinePoints)
        {
            allPointsTemp.Add(point);
            allPointsTemp.Add(new Vector3(point.x + (float)(depth * xDir), point.y + (float)(depth * yDir), point.z + (float)(depth * zDir)));
        }
        foreach (Vector3 point in polyLinePoints)
        {
            allPointsTemp.Add(new Vector3(point.x + (float)(depth * xDir), point.y + (float)(depth * yDir), point.z + (float)(depth * zDir)));
        }

        allPoints = new List<Vector3>();
        foreach (Vector3 point in allPointsTemp)
        {
            Vector4 vect4Point = new Vector4(point.x, point.y, point.z, 1);
            Vector3 tempVect3Point = matrixTrans * vect4Point;
            allPoints.Add(tempVect3Point);
        }

        triangleNums = CreateTriangles(polyLinePoints);
    }

    private static List<int> CreateTriangles(List<Vector3> polyLinePoints)
    {
        if (polyLinePoints.Count == 0)
        {
            return new List<int>();
        }

        List<int> triangleNums = new List<int>();
        int mainPointCount = polyLinePoints.Count;
        int mainPointCountDouble = mainPointCount * 2;
        int mainPointCountTriple = mainPointCount * 3;

        triangleNums.AddRange(EarClippingVariant(polyLinePoints, FaceSide.FRONT));
        List<int> topTriangles = EarClippingVariant(polyLinePoints, FaceSide.BACK);
        foreach (int tri in topTriangles)
        {
            triangleNums.Add(tri + mainPointCountTriple);
        }
        //for (int i = 0; i < mainPointCount - 2; i++)
        //{
        //    triangleNums.Add(0);
        //    triangleNums.Add(i + 2);
        //    triangleNums.Add(i + 1);

        //    triangleNums.Add(mainPointCountTriple);
        //    triangleNums.Add(mainPointCountTriple + i + 1);
        //    triangleNums.Add(mainPointCountTriple + i + 2);
        //}
        for (int i = 0; i < mainPointCountDouble; i++)
        {
            if (i % 2 == 0)
            {
                triangleNums.Add(i + mainPointCount);
                triangleNums.Add((i + 2) % mainPointCountDouble + mainPointCount);
                triangleNums.Add((i + 1) % mainPointCountDouble + mainPointCount);
            }
            else
            {
                triangleNums.Add(i + mainPointCount);
                triangleNums.Add((i + 1) % mainPointCountDouble + mainPointCount);
                triangleNums.Add((i + 2) % mainPointCountDouble + mainPointCount);
            }
        }

        return triangleNums;
    }

    private static List<Vector3> GetExtrudeFacePoints(IfcProfileDef profDef)
    {
        List<Vector3> polyLinePoints = new List<Vector3>();
        if (profDef.ProfileType == IfcProfileTypeEnum.AREA)
        {
            IfcRectangleProfileDef rectangleProfileDef = profDef as IfcRectangleProfileDef;
            if (rectangleProfileDef != null)
            {
                double xDim = rectangleProfileDef.XDim;
                double yDim = rectangleProfileDef.YDim;
                IfcAxis2Placement2D posit = rectangleProfileDef.Position;
                Matrix4x4 matrixTrans2D = ExtractMatixAxis2Placement2D(posit);

                List<Vector4> pointListTemp = new List<Vector4>();
                pointListTemp.Add(new Vector4(-(float)xDim / 2.0f, (float)yDim / 2.0f, 0, 1));
                pointListTemp.Add(new Vector4((float)xDim / 2.0f, (float)yDim / 2.0f, 0, 1));
                pointListTemp.Add(new Vector4((float)xDim / 2.0f, -(float)yDim / 2.0f, 0, 1));
                pointListTemp.Add(new Vector4(-(float)xDim / 2.0f, -(float)yDim / 2.0f, 0, 1));
                pointListTemp.Add(new Vector4(-(float)xDim / 2.0f, (float)yDim / 2.0f, 0, 1));

                foreach (Vector4 poi in pointListTemp)
                {
                    Vector4 pointTemp = matrixTrans2D * poi;
                    polyLinePoints.Add(new Vector3(pointTemp.x, pointTemp.y, pointTemp.z));
                }
            }

            IfcArbitraryClosedProfileDef arbitraryClosedProfileDef = profDef as IfcArbitraryClosedProfileDef;
            if (arbitraryClosedProfileDef != null)
            {
                IfcCompositeCurve comCurve = arbitraryClosedProfileDef.OuterCurve as IfcCompositeCurve;
                if (comCurve != null)
                {
                    foreach (IfcCompositeCurveSegment compCurveSef in comCurve.Segments)
                    {
                        IfcPolyline polyLineComp = compCurveSef.ParentCurve as IfcPolyline;
                        if (polyLineComp != null)
                        {
                            foreach (IfcCartesianPoint point in polyLineComp.Points)
                            {
                                Vector3 newPoint = new Vector3((float)point.Coordinates.Item1, (float)point.Coordinates.Item2, (float)point.Coordinates.Item3);
                                polyLinePoints.Add(newPoint);
                            }
                        }

                        IfcTrimmedCurve trimmedCurveComp = compCurveSef.ParentCurve as IfcTrimmedCurve;
                        if (trimmedCurveComp != null)
                        {
                            // TODO: Add this if it is ever used

                        }
                    }
                }

                IfcPolyline polyLine = arbitraryClosedProfileDef.OuterCurve as IfcPolyline;
                if (polyLine != null)
                {
                    foreach (IfcCartesianPoint point in polyLine.Points)
                    {
                        Vector3 newPoint = new Vector3((float)point.Coordinates.Item1, (float)point.Coordinates.Item2, (float)point.Coordinates.Item3);
                        polyLinePoints.Add(newPoint);
                    }
                }
            }
        }
        if (profDef.ProfileType == IfcProfileTypeEnum.CURVE)
        {
            // TODO: Add this if it is ever used

        }

        return polyLinePoints;
    }

    private static void GetTrianglesFromPolyLine(IfcPolyline polyLine, out List<Vector3> polyLinePoints, out List<int> triangleNums, FaceSide faceSide)
    {
        polyLinePoints = new List<Vector3>();
        if (polyLine != null)
        {
            foreach (IfcCartesianPoint point in polyLine.Points)
            {
                Vector3 newPoint = new Vector3((float)point.Coordinates.Item1, (float)point.Coordinates.Item2, (float)point.Coordinates.Item3);
                polyLinePoints.Add(newPoint);
            }
        }

        triangleNums = new List<int>();
        for (int i = 0; i < polyLinePoints.Count - 2; i++)
        {
            if (faceSide == FaceSide.FRONT || faceSide == FaceSide.BOTH)
            {
                triangleNums.Add(0);
                triangleNums.Add(i + 2);
                triangleNums.Add(i + 1);
            }

            if (faceSide == FaceSide.BACK || faceSide == FaceSide.BOTH)
            {
                triangleNums.Add(0);
                triangleNums.Add(i + 1);
                triangleNums.Add(i + 2);
            }
        }
    }

    private static void GetTriangleFromClippedItem(IfcBooleanClippingResult item, out List<Vector3> allPoints, out List<int> triangleNums)
    {
        allPoints = new List<Vector3>();
        triangleNums = new List<int>();

        IfcRepresentationItem firstOpItem = item.FirstOperand as IfcRepresentationItem;
        List<Vector3> allPointsFirstOpItem = new List<Vector3>();
        List<int> triangleNumsFirstOpItem = new List<int>();
        GetTriangleFromOperand(firstOpItem, ref allPointsFirstOpItem, ref triangleNumsFirstOpItem);

        IfcRepresentationItem secondOpItem = item.FirstOperand as IfcRepresentationItem;
        List<Vector3> allPointsSecondOpItem = new List<Vector3>();
        List<int> triangleNumsSecondOpItem = new List<int>();
        GetTriangleFromOperand(secondOpItem, ref allPointsSecondOpItem, ref triangleNumsSecondOpItem);

        // TODO: Need to use the operation rather than just return the UNION
        IfcBooleanOperator op = item.Operator;
        if (op == IfcBooleanOperator.DIFFERENCE)
        {

        }
        if (op == IfcBooleanOperator.UNION)
        {

        }
        if (op == IfcBooleanOperator.INTERSECTION)
        {

        }

        allPoints.AddRange(allPointsFirstOpItem);
        triangleNums.AddRange(triangleNumsFirstOpItem);

        allPoints.AddRange(allPointsSecondOpItem);
        int allPointsFirstOpItemCount = allPointsFirstOpItem.Count;
        triangleNumsSecondOpItem.ForEach(tN => tN += allPointsFirstOpItemCount);
        triangleNums.AddRange(triangleNumsSecondOpItem);
    }

    private static void GetTriangleFromOperand(IfcRepresentationItem opItem, ref List<Vector3> allPoints, ref List<int> triangleNums)
    {
        if (opItem.GetType() == typeof(IfcBooleanClippingResult))
        {
            GetTriangleFromClippedItem(opItem as IfcBooleanClippingResult, out allPoints, out triangleNums);
        }
        if (opItem.GetType() == typeof(IfcExtrudedAreaSolid))
        {
            GetTrianglesForExtrudedSolid(opItem as IfcExtrudedAreaSolid, out allPoints, out triangleNums);
        }
        if (opItem.GetType() == typeof(IfcPolygonalBoundedHalfSpace))
        {
            // TODO: Figure out what this is:
        }
    }

    public static Matrix4x4 ExtractMatixAxis2Placement2D(IfcAxis2Placement2D posit)
    {
        IfcCartesianPoint loc = posit.Location;
        IfcDirection refDir = posit.RefDirection;
        Vector4 locVec = new Vector4((float)loc.Coordinates.Item1, (float)loc.Coordinates.Item2, 0, 1);
        Vector4 refDirVec = new Vector4((float)refDir.DirectionRatioX, (float)refDir.DirectionRatioY, 0, 0);
        Vector4 axisVec = new Vector4(0, 0, 1, 0);
        Vector4 otherAxis = Vector3.Cross(axisVec, refDirVec);
        Matrix4x4 matrixTrans = new Matrix4x4(refDirVec, otherAxis, axisVec, locVec);
        return matrixTrans;
    }

    public static Matrix4x4 ExtracMatrixAxis2Placement3D(IfcAxis2Placement3D axisPlace)
    {
        IfcCartesianPoint loc = axisPlace.Location;
        IfcDirection axis = axisPlace.Axis;
        IfcDirection refDir = axisPlace.RefDirection;
        Vector4 locVec = new Vector4((float)loc.Coordinates.Item1, (float)loc.Coordinates.Item2, (float)loc.Coordinates.Item3, 1);
        Vector4 axisVec = (axis != null) ?
            new Vector4((float)axis.DirectionRatioX, (float)axis.DirectionRatioY, (float)axis.DirectionRatioZ, 0)
            : new Vector4(0, 0, 1, 0);
        Vector4 refDirVec = (refDir != null) ?
            new Vector4((float)refDir.DirectionRatioX, (float)refDir.DirectionRatioY, (float)refDir.DirectionRatioZ, 0)
            : new Vector4(1, 0, 0, 0);
        Vector4 otherAxis = Vector3.Cross(axisVec, refDirVec);
        Matrix4x4 matrixTrans = new Matrix4x4(refDirVec, otherAxis, axisVec, locVec);
        return matrixTrans;
    }

    public static void CreateTrianglesFromFaceList(ref List<Vector3> newVertices, ref List<int> newTriangles, List<IfcFace> faces, FaceSide faceSide)
    {
        foreach (IfcFace face in faces)
        {
            List<IfcFaceBound> bounds = face.Bounds.ToList();
            foreach (IfcFaceBound bound in bounds)
            {
                IfcPolyloop loop = bound.Bound as IfcPolyloop;
                List<IfcCartesianPoint> points = loop.Polygon.ToList();
                List<Vector3> pointsVects = new List<Vector3>();
                foreach (IfcCartesianPoint point in points)
                {
                    pointsVects.Add(new Vector3((float)point.Coordinates.Item1, (float)point.Coordinates.Item2, (float)point.Coordinates.Item3));
                }

                // Rather then just (vect0 - vecti - vectj) triangles I used earclipping (polygon trianglulation)
                if (pointsVects.Count >= 3)
                {
                    //List<int> tempTrianlgeInts = new List<int>();
                    //for (int index = 0; index < pointsVects.Count - 2; index++)
                    //{
                    //    tempTrianlgeInts.Add(0);
                    //    tempTrianlgeInts.Add(index + 2);
                    //    tempTrianlgeInts.Add(index + 1);
                    //}

                    List<int> tempTrianlgeInts = EarClippingVariant(pointsVects, FaceSide.FRONT);

                    int verticiesCount = newVertices.Count;
                    foreach (int intVal in tempTrianlgeInts)
                    {
                        newTriangles.Add(intVal + verticiesCount);
                    }
                    newVertices.AddRange(pointsVects);
                }
            }
        }
    }

    private static List<int> EarClippingVariant(List<Vector3> points, FaceSide faceSide)
    {
        List<int> triangleList = new List<int>();

        List<Tuple<int, Vector3>> remainingPoints = new List<Tuple<int, Vector3>>();
        for (int i = 0; i < points.Count; i++)
        {
            Vector3 p = points[i];
            remainingPoints.Add(new Tuple<int, Vector3>(i, p));
        }

        while (remainingPoints.Count > 3)
        {
            int pCount = remainingPoints.Count;
            bool removedEar = false;
            for (int i = 0; i < pCount; i++)
            {
                int index2 = (i + 1) % pCount;
                int index3 = (i + 2) % pCount;
                Vector3 p1 = remainingPoints[i].Item2;
                Vector3 p2 = remainingPoints[index2].Item2;
                Vector3 p3 = remainingPoints[index3].Item2;

                bool inTriangle = false;
                for (int j = 0; j < remainingPoints.Count; j++)
                {
                    if ((j == i) || (j == (i + 1)) || (j == (i + 2)))
                    {
                        continue;
                    }
                    Vector3 p4 = remainingPoints[j].Item2;
                    if (PointInTriangle(p4, p1, p2, p3))
                    {
                        inTriangle = true;
                        break;
                    }
                }
                if (!inTriangle)
                {
                    if (faceSide == FaceSide.FRONT)
                    {
                        triangleList.Add(remainingPoints[i].Item1);
                        triangleList.Add(remainingPoints[index3].Item1);
                        triangleList.Add(remainingPoints[index2].Item1);
                    }
                    if (faceSide == FaceSide.BACK)
                    {
                        triangleList.Add(remainingPoints[i].Item1);
                        triangleList.Add(remainingPoints[index2].Item1);
                        triangleList.Add(remainingPoints[index3].Item1);
                    }
                    remainingPoints.RemoveAt(index2);
                    removedEar = true;
                    break;
                }
            }

            if (!removedEar)
            {
                break;
            }
        }

        if (faceSide == FaceSide.FRONT || faceSide == FaceSide.BOTH)
        {
            triangleList.Add(remainingPoints[0].Item1);
            triangleList.Add(remainingPoints[2].Item1);
            triangleList.Add(remainingPoints[1].Item1);
        }
        if (faceSide == FaceSide.BACK || faceSide == FaceSide.BOTH)
        {
            triangleList.Add(remainingPoints[0].Item1);
            triangleList.Add(remainingPoints[1].Item1);
            triangleList.Add(remainingPoints[2].Item1);
        }

        return triangleList;
    }

    private static bool SameSide(Vector3 p1, Vector3 p2, Vector3 a, Vector3 b)
    {
        Vector3 cp1 = Vector3.Cross(b - a, p1 - a);
        Vector3 cp2 = Vector3.Cross(b - a, p2 - a);
        if (Vector3.Dot(cp1, cp2) >= 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private static bool PointInTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        if (SameSide(p, a, b, c) && SameSide(p, b, a, c) && SameSide(p, c, a, b))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    public static GameObject CreateBoundingBoxCube(BoundingBox boundingBox)
    {

            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);

        try
        {
            cube.transform.position = boundingBox.Center;
        cube.transform.localScale = boundingBox.Dimentions;

        //float angleX = Vector3.Angle(boundingBox.RightDirection, Vector3.right);
        //float angleY = Vector3.Angle(boundingBox.UpDirection, Vector3.up);
        //float angleZ = Vector3.Angle(boundingBox.ForwardDirection, Vector3.forward);
        //cube.transform.Rotate(new Vector3(angleX, angleY, angleZ));

        Vector3 X = boundingBox.RightDirection;
        Vector3 Y = boundingBox.ForwardDirection;
        Vector3 Z = boundingBox.UpDirection;
        float alpha = Mathf.Atan2(-Z.y, Z.z);
        float beta = (Mathf.Asin(Z.x));
        float gamma = -Mathf.Atan2(-Y.x, X.x);

            cube.transform.Rotate(new Vector3(alpha * 180 / Mathf.PI, beta * 180 / Mathf.PI, gamma * 180 / Mathf.PI));
        }
        catch { }

        return cube;
    }

    public static void GetModelCenterAndDims(GameObject go, out Vector3 center, out Vector3 diment)
    {
        Bounds b = new Bounds(go.transform.position, Vector3.zero);
        foreach (Renderer r in go.GetComponentsInChildren<Renderer>())
        {
            b.Encapsulate(r.bounds);
        }

        center = b.center;
        diment = b.size;
    }

    public static IfcElement CreateIfcElement(DatabaseIfc db, GameObject ifcGameObject, out List<Vector3> newVertices, float scale)
    {
        Vector3 goPos = ifcGameObject.transform.position;

        // Each Child that has a mesh should be a seperate IfcRepresentationItem but the location is relative to the parent (unless we can just use global coordinates?)
        List<IfcRepresentationItem> repItems = new List<IfcRepresentationItem>();
        newVertices = new List<Vector3>();
        GetShapRepRecusive(db, ifcGameObject, ref repItems, ref newVertices, goPos, scale);

        IfcShapeRepresentation newShapeRep = new IfcShapeRepresentation(repItems);
        newShapeRep.RepresentationType = "Brep";
        IfcProductDefinitionShape productDefShape = new IfcProductDefinitionShape(newShapeRep);

        Vector3 objLocation = ifcGameObject.transform.position;
        IfcCartesianPoint objPoint = new IfcCartesianPoint(db, objLocation.x, objLocation.z, objLocation.y);
        IfcAxis2Placement3D objPlace3D = new IfcAxis2Placement3D(objPoint);
        IfcLocalPlacement localPlace = new IfcLocalPlacement(objPlace3D);

        IfcFurnishingElement newElem = new IfcFurnishingElement(db.Project, localPlace, productDefShape);

        return newElem as IfcElement;
    }

    private static void GetShapRepRecusive(DatabaseIfc db, GameObject go, ref List<IfcRepresentationItem> repItems, ref List<Vector3> newVertices, Vector3 goPos, float scale)
    {
        MeshFilter meshFilter = go.GetComponent<MeshFilter>();
        if (meshFilter != null)
        {
            List<IfcFace> faces = new List<IfcFace>();
            List<Vector3> meshVerticies = meshFilter.mesh.vertices.ToList();
            List<int> meshTriangles = meshFilter.mesh.triangles.ToList();

            for (int tri = 0; tri < meshTriangles.Count; tri += 3)
            {
                List<IfcCartesianPoint> points = new List<IfcCartesianPoint>();
                Vector3 p1 = meshVerticies[meshTriangles[tri]];
                Vector3 p2 = meshVerticies[meshTriangles[tri + 1]];
                Vector3 p3 = meshVerticies[meshTriangles[tri + 2]];

                // Use global location
                Vector3 p1Trans = go.transform.TransformPoint(p1);
                Vector3 p2Trans = go.transform.TransformPoint(p2);
                Vector3 p3Trans = go.transform.TransformPoint(p3);

                // scale
                p1Trans = new Vector3(p1Trans.x / scale, p1Trans.y / scale, p1Trans.z / scale);
                p2Trans = new Vector3(p2Trans.x / scale, p2Trans.y / scale, p2Trans.z / scale);
                p3Trans = new Vector3(p3Trans.x / scale, p3Trans.y / scale, p3Trans.z / scale);

                newVertices.AddRange(new List<Vector3>() { p1Trans, p2Trans, p3Trans });

                // IFC has flipped Z and Y...
                points.Add(new IfcCartesianPoint(db, p1Trans.x, p1Trans.z, p1Trans.y));
                points.Add(new IfcCartesianPoint(db, p2Trans.x, p2Trans.z, p2Trans.y));
                points.Add(new IfcCartesianPoint(db, p3Trans.x, p3Trans.z, p3Trans.y));

                IfcPolyloop loop = new IfcPolyloop(points);
                IfcFaceBound bound = new IfcFaceBound(loop, true);
                List<IfcFaceBound> bounds = new List<IfcFaceBound>() { bound };
                IfcFace face = new IfcFace(bounds);
                faces.Add(face);
            }

            IfcClosedShell closedShell = new IfcClosedShell(faces);
            IfcFacetedBrep facetedBrep = new IfcFacetedBrep(closedShell);
            repItems.Add(facetedBrep);
        }

        for (int i = 0; i < go.transform.childCount; i++)
        {
            GameObject child = go.transform.GetChild(i).gameObject;
            GetShapRepRecusive(db, child, ref repItems, ref newVertices, goPos, scale);
        }
    }

    public static void GetMeshVerticiesList(GameObject go, ref List<Vector3> vector3s)
    {
        if (go != null)
        {
            MeshFilter meshFilter = go.GetComponent<MeshFilter>();
            if (meshFilter != null)
            {
                //vector3s.AddRange(meshFilter.mesh.vertices);
                foreach (Vector3 v in meshFilter.mesh.vertices)
                {
                    Vector3 worldPt = go.transform.TransformPoint(v);
                    vector3s.Add(worldPt);
                }
            }
            for (int i = 0; i < go.transform.childCount; i++)
            {
                GetMeshVerticiesList(go.transform.GetChild(i).gameObject, ref vector3s);
            }
        }
    }







    #region EXTRA ==================================================================================================================

    private void CreateObjectSimple(DatabaseIfc db, IfcProject project, double xLoc, double yLoc, double zLoc, double length, double width, double height, double angle, string nameId)
    {
        // location line of object
        //IfcCartesianPoint p1 = new IfcCartesianPoint(db, 0, 0, 0);
        //IfcCartesianPoint p2 = new IfcCartesianPoint(db, length, width, height);
        //IfcPolyline pLine = new IfcPolyline(p1, p2);
        //IfcShapeRepresentation repLine = IfcShapeRepresentation.GetAxisRep(pLine);

        // Create the shape based on the wall properties
        IfcCartesianPoint p3 = new IfcCartesianPoint(db, 0, 0, 0);
        IfcAxis2Placement3D place1 = new IfcAxis2Placement3D(p3);
        IfcRectangleProfileDef rectProf = new IfcRectangleProfileDef(db, "rect", length, width);
        IfcExtrudedAreaSolid extrudeArea = new IfcExtrudedAreaSolid(rectProf, place1, height);
        IfcShapeRepresentation newShapeRep = new IfcShapeRepresentation(extrudeArea);

        // Create the furnishing element:
        IfcCartesianPoint p4 = new IfcCartesianPoint(db, xLoc, yLoc, zLoc);
        IfcAxis2Placement3D place2 = new IfcAxis2Placement3D(p4);
        IfcDirection refDir = new IfcDirection(db, Math.Cos(angle * Math.PI / 180.0), Math.Sin(angle * Math.PI / 180.0), 0);
        place2.RefDirection = refDir;
        IfcLocalPlacement localPlace = new IfcLocalPlacement(place2);
        List<IfcShapeModel> reps = new List<IfcShapeModel>();
        //reps.Add(repLine);
        reps.Add(newShapeRep);
        IfcProductDefinitionShape newElemReps = new IfcProductDefinitionShape(reps);
        IfcFurnishingElement newElem = new IfcFurnishingElement(project, localPlace, newElemReps);
        newElem.Name = nameId;
    }

    #endregion
}