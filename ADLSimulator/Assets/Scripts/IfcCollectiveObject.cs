using GeometryGym.Ifc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public class BoundingBox
{
    public Vector3 Center;
    public Vector3 Dimentions;
    public Vector3 ForwardDirection;
    public Vector3 UpDirection;
    public Vector3 RightDirection;

    public BoundingBox(Vector3 center, Vector3 dimentions, Vector3 forwardDirection, Vector3 upDirection, Vector3 rightDirection)
    {
        Center = center;
        Dimentions = dimentions;
        ForwardDirection = forwardDirection;
        UpDirection = upDirection;
        RightDirection = rightDirection;
    }
}

public class Property
{
    public string PropertyName;
    public object PropertyValue;

    public Property(string name, object value)
    {
        PropertyName = name;
        PropertyValue = value;
    }

    public static bool AreEqual(Property obj1, Property obj2)
    {
        return (obj1.PropertyName == obj2.PropertyName && obj1.PropertyValue == obj2.PropertyValue);
    }

    public string GetString()
    {
        return PropertyName + "=" + PropertyValue;
    }

    public static List<Property> ParsePropertyString(string propertyString)
    {
        string[] stringArray = propertyString.Split(';');
        List<Property> props = new List<Property>();
        foreach (string s in stringArray)
        {
            string[] propSplit = s.Split('=');
            if (propSplit.Length == 2)
            {
                props.Add(new Property(propSplit[0], propSplit[1]));
            }
        }

        return props;
    }
}

public class IfcVirtualObject
{
    public IfcVirtualObject()
    {

    }
}

public class IfcCollectiveObject
{
    // Must have properties:
    public string Name { get; private set; }
    public Guid Id { get; private set; }
    public string IfcType { get; private set; }

    // Geometry Stuff:
    public GameObject IfcGameObject { get; private set; }
    public GameObject GameObjectBB { get; private set; }
    public List<Vector3> GlobalVerticies { get; private set; }
    public Vector3 Direction { get; private set; }
    public BoundingBox BoundingBox { get; private set; }
    public float Scale { get; private set; }

    // IFC stuff:
    public IfcElement IfcElem { get; private set; }
    public Material MainMat { get; private set; }
    public List<Property> Properties { get; private set; }

    //Constructors:

    public IfcCollectiveObject(IfcElement newIfcElem, float scale)
    {
        IfcElem = newIfcElem;
        Scale = scale;

        //Standard Properties:
        Name = IfcElem.Name;
        Id = IfcElem.Guid;
        IfcType = IfcElem.GetType().ToString();

        // Create new objects:
        IfcGameObject = new GameObject(Name);

        // Add mess info:
        IfcGameObject.AddComponent<MeshFilter>();
        IfcGameObject.AddComponent<MeshRenderer>();
        Mesh mesh = new Mesh();
        List<Vector3> newVertices = new List<Vector3>();
        List<int> newTriangles = new List<int>();

        // Get the triangle and Mesh info:
        IfcProductDefinitionShape shapeRep = IfcElem.Representation as IfcProductDefinitionShape;
        List<IfcShapeModel> shapeModels = shapeRep.Representations.ToList();
        foreach (IfcShapeRepresentation shapeRepModel in shapeModels)
        {
            if (shapeRepModel.RepresentationIdentifier == "Body")
            {
                IfcMeshUtils.GetTrianglesFromShapeRepresentation(ref newVertices, ref newTriangles, shapeRepModel);
            }
        }

        // Scale All vertices:
        List<Vector3> scaledVertices = new List<Vector3>();
        foreach (Vector3 v in newVertices)
        {
            scaledVertices.Add(new Vector3(v.x * Scale, v.y * Scale, v.z * Scale));
        }
        newVertices = scaledVertices;

        BoundingBox boundingBox = null;
        List<Vector3> newVerticesFinal = IfcMeshUtils.PositionElementAndGetBB(IfcElem, newVertices, out boundingBox, Scale);

        // Assign mesh and materials:
        if (newVerticesFinal != null)
        {
            IfcGameObject.GetComponent<MeshFilter>().mesh = mesh;
            mesh.SetVertices(newVerticesFinal);
            mesh.SetTriangles(newTriangles.ToArray(), 0);
            Vector2[] uvs = IfcMeshUtils.CalculateUVs(mesh, newVerticesFinal);
            mesh.SetUVs(0, new List<Vector2>(uvs));
            mesh.RecalculateNormals();
        }

        GlobalVerticies = newVerticesFinal;

        if (boundingBox != null)
        {
            GameObjectBB = IfcMeshUtils.CreateBoundingBoxCube(boundingBox);
            GameObjectBB.name = "cube" + Name;
            BoundingBox = boundingBox;
        }

        SetIfcProperties();
    }

    public IfcCollectiveObject(GameObject newGameObject, DatabaseIfc db, float scale)
    {
        // Get mesh info:
        IfcGameObject = newGameObject;
        Name = newGameObject.name;
        MainMat = FindMainMat(IfcGameObject);

        Scale = scale;

        // Create IFC object:
        List<Vector3> newVertices = new List<Vector3>();
        IfcElem = IfcMeshUtils.CreateIfcElement(db, IfcGameObject, out newVertices, scale);

        // Standard Properties:
        IfcElem.Name = newGameObject.name;
        Name = IfcElem.Name;
        Id = IfcElem.Guid;
        IfcType = IfcElem.GetType().ToString();

        newVertices = new List<Vector3>();
        IfcMeshUtils.GetMeshVerticiesList(IfcGameObject, ref newVertices);

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

        BoundingBox boundingBox = new BoundingBox(centerV, dimsV, centerVforward - centerV, centerVup - centerV, centerVright - centerV);

        GlobalVerticies = newVertices;

        if (boundingBox != null)
        {
            //Debug.Log("CUBEEEEEEEEEEEEEE");
            GameObjectBB = IfcMeshUtils.CreateBoundingBoxCube(boundingBox);
            GameObjectBB.name = "cube" + Name;
            BoundingBox = boundingBox;
        }

        SetIfcProperties();
    }

    // Methods:

    private Material FindMainMat(GameObject go)
    {
        MeshRenderer mr = go.GetComponent<MeshRenderer>();
        if (mr != null)
        {
            return mr.material;
        }
        for (int i = 0; i < go.transform.childCount; i++)
        {
            Material childMat = FindMainMat(go.transform.GetChild(i).gameObject);
            if (childMat != null)
            {
                return childMat;
            }
        }
        return null;
    }

    public void SetMaterial(Material mat)
    {
        SetMeshRenderMaterial(GameObjectBB, mat);
        SetMeshRenderMaterial(IfcGameObject, mat);
        MainMat = mat;
    }

    public string GetInfoText()
    {
        string messageString = "";
        try
        {
            messageString += "\n" + "Name: " + Name;
            messageString += "\n" + "GlobalId: " + Id.ToString();
            messageString += "\n" + "IfcType: " + IfcType;

            messageString += "\n";
            foreach (Property prop in Properties)
            {
                messageString += "\n" + prop.PropertyName + ": " + prop.PropertyValue.ToString();
            }

        }
        catch (Exception ex)
        {
            Debug.Log(ex.ToString());
        }

        return messageString;
    }

    public void Highlightobjects(Material highlightMat)
    {
        SetMeshRenderMaterial(GameObjectBB, highlightMat);
        SetMeshRenderMaterial(IfcGameObject, highlightMat);
    }

    public void Unhighlightobjects()
    {
        SetMeshRenderMaterial(GameObjectBB, MainMat);
        SetMeshRenderMaterial(IfcGameObject, MainMat);
    }

    public void ToggleMeshRenderer(bool boundingBoxActive)
    {
        ToggleMeshRenderActive(GameObjectBB, boundingBoxActive);
        ToggleMeshRenderActive(IfcGameObject, !boundingBoxActive);
    }

    public void ToggleMeshRenderActive(GameObject go, bool active)
    {
        if (go != null)
        {
            for (int i = 0; i < go.transform.childCount; i++)
            {
                ToggleMeshRenderActive(go.transform.GetChild(i).gameObject, active);
            }

            Renderer rend = go.GetComponent<Renderer>();
            if (rend != null)
            {
                rend.enabled = active;
            }
        }
    }

    public void DeactivateObjects()
    {
        
        if (GameObjectBB != null)
            GameObjectBB.SetActive(false);
        if (IfcGameObject != null)
            IfcGameObject.SetActive(false);
        
        if (GameObjectBB != null)
            GameObject.Destroy(GameObjectBB);
        
        if (IfcGameObject != null)
            GameObject.Destroy(IfcGameObject);
        
    }

    public void SetMeshRenderMaterial(GameObject go, Material mat)
    {
        if (go != null)
        {
            MeshRenderer mr = go.GetComponent<MeshRenderer>();
            if (mr != null)
            {
                mr.material = mat;
            }
            for (int i = 0; i < go.transform.childCount; i++)
            {
                SetMeshRenderMaterial(go.transform.GetChild(i).gameObject, mat);
            }
        }
    }

    private void SetIfcProperties()
    {
        Properties = new List<Property>();

        // TODO: Other properties such as description should go here:



        // These are really only the properties of the object Type not of the individual object
        List<IfcRelDefinesByProperties> relsDefineByProperties = IfcElem.IsDefinedBy.ToList();
        foreach (IfcRelDefinesByProperties reDefByProp in relsDefineByProperties)
        {
            IfcPropertySet propertySet = reDefByProp.RelatingPropertyDefinition as IfcPropertySet;
            if (propertySet != null)
            {
                foreach (KeyValuePair<string, IfcProperty> property in propertySet.HasProperties)
                {
                    string propName = property.Key;
                    IfcPropertySingleValue propertySingleVal = property.Value as IfcPropertySingleValue;
                    if (propertySingleVal != null)
                    {
                        Property newProperty = new Property(propName, propertySingleVal.NominalValue.Value);
                        Properties.Add(newProperty);
                    }
                }
            }
        }
    }

    public void AddElementProperty(IfcElement elem, DatabaseIfc db, string propertyName, string propertyValue)
    {
        Properties.Add(new Property(propertyName, propertyValue));

        
        // TODO: add it the the IFC database too:

        IReadOnlyCollection<IfcRelDefinesByProperties> reldefs = elem.IsDefinedBy;
        if (reldefs.Count == 0)
        {
            IfcPropertySet ips = new IfcPropertySet(db, elem.Name + "_actions");
            IfcRelDefinesByProperties reldef = new IfcRelDefinesByProperties(elem, ips);
            reldefs = elem.IsDefinedBy;
        }

        //Dictionary<string, IfcProperty> properties = new Dictionary<string, IfcProperty>();

        foreach (IfcRelDefinesByProperties reldef in reldefs)
        {
            string name = reldef.Name;
            IfcPropertySetDefinition propdef = reldef.RelatingPropertyDefinition;
            IfcPropertyDefinition pd = propdef as IfcPropertyDefinition;
            IfcPropertySet ps = pd as IfcPropertySet;

            // I am just adding it here to the first propertyset it finds. You may want to find a specific one.
            // But since the GetElementProperties gets all together maybe who cares right?
            ps.AddProperty(new IfcPropertySingleValue(db, propertyName, propertyValue));
            ///Properties.Remove(elem);
            break;
        }
    }

    public Dictionary<string, IfcProperty> GetElementProperties(IfcElement elem)
    {
        // May want to do null checks trhoughout and debug to find out if there are other types used.
        // IFC will often throw curveballs where some other type is being used depending on the BIM App
        // that created the model; test and debug with alot of models and add the code each time...

        // Also, Properties can have the same name in the model so may want to handle that somehow
        // Either don't use a dictionary and make it a list of property tuples or don't allow duplicate properties...

        IReadOnlyCollection<IfcRelDefinesByProperties> reldefs = elem.IsDefinedBy;
        Dictionary<string, IfcProperty> properties = new Dictionary<string, IfcProperty>();
        foreach (IfcRelDefinesByProperties reldef in reldefs)
        {
            string name = reldef.Name;
            IfcPropertySetDefinition propdef = reldef.RelatingPropertyDefinition;
            IfcPropertyDefinition pd = propdef as IfcPropertyDefinition;
            IfcPropertySet ps = pd as IfcPropertySet;
            foreach (KeyValuePair<string, IfcProperty> kvp in ps.HasProperties)
            {
                string propertyName = kvp.Key;
                IfcProperty propertyValue = kvp.Value;
                IfcPropertySingleValue psv = propertyValue as IfcPropertySingleValue;

                string valString = psv.NominalValue.Value.ToString();

                if (!properties.ContainsKey(propertyName))
                {
                    properties.Add(propertyName, propertyValue);
                }
            }
        }

        return properties;
    }

    public string GetLongStringThatMayContainMaterial()
    {
        string returnString = "";
        try
        {
            returnString += Name ?? "";
            returnString += IfcElem.Description ?? "";
            returnString += String.Join(String.Empty, IfcElem.Comments) ?? "";
            returnString += IfcElem.Tag ?? "";
            foreach (Property p in Properties)
            {
                returnString += p.PropertyValue.ToString() ?? "";
            }
            returnString += IfcElem.MaterialSelect.Name ?? "";
        }
        catch
        {
        }

        return returnString;
    }   

}