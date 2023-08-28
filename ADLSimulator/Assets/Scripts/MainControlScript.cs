using GeometryGym.Ifc;
using IfcEngineWrapper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.AI;

#if _WIN64
using int_t = System.Int64;
#else
using int_t = System.Int32;
#endif

public class MainControlScript : MonoBehaviour
{
    #region Materials:
    public Material defaultMat1;
    public Material defaultMat2;
    public Material defaultMat3;
    public Material defaultMat4;
    public Material highlightMatYellow;
    public Material highlightMatRed;
    #endregion

    // Most of these can be set at the start using ID/name and made private but meh this is the same:
    #region UI Items:
    public Toggle ToggleObjects;
    public Button DeleteObjects;
    public Slider TopSlider;
    public Slider BottomSlider;
    public Slider ZoomSlider;
    public Text ObjectInfo;
    public InputField ExportFileName;
    public GameObject ObjectSelectionContent;
    public Button Quit;
    public Button Export;
    public Button AddProperty;
    public Button doneParameter;
    public Canvas MainCanvas;
    public Canvas AddParamCanvas;
    public InputField ParamNameText;
    public InputField ParamValueText;
    #endregion

    #region IFC Items
    private string ifcModelDirectory;
    public DatabaseIfc IfcDataBase;
    private GameObject rootGameObject;
    private GameObject cubesGameObject;
    private Dictionary<string, List<IfcCollectiveObject>> goDictionary;
    private IfcCollectiveObject selectedObject;
    private List<IfcCollectiveObject> ruleInstanceHighightedObjects;
    private GameObject gameObjBeingAdded;
    private List<Tuple<GameObject, Sprite>> objectTuples;
    #endregion

    #region Control Items
    public Camera mainCamera;
    private Ray ray;
    private RaycastHit hit;
    private bool objectLocked;
    private bool objectBeingAdded;
    private float modelScale;
    private float cameraClipPlane;
    private float cameraSize;
    #endregion

    #region Rules Items
    public static float INCH_TO_METER = 0.0254f;
    public static float FT_TO_METER = 0.3048f;
    public static float CM_TO_METER = 0.01f;
    public static float MM_TO_METER = 0.001f;
    #endregion

    #region Model, Agent, and Task Planning
    private string modelName;
    private string agentName;
    private string sessionName;

    private float timeScale;
    private float simulationLengthInMinutes;

    public List<string> agentTypeNames;
    public List<string> agentTaskFileNames;
    private Dictionary<string, int_t> agentTypeAndIDDictionary;
    private List<string> distinctAgentTypeNames;
    private List<Agent> agentArray;

    private TaskPlanner taskPlanner;
    private float endSimulationTime;
    string dir = "Assets\\Logs\\" + DateTime.Now.ToString("yyyy-MM-dd_hh-mm-ss") + "\\";

    #endregion

    // Use this for initialization
    void Start()
    {
        Application.runInBackground = true;
        //TODO remove this line after debugging
        Application.stackTraceLogType = StackTraceLogType.None;

        // Get information for the Editor only
        modelName = SimulationConfiguration.ModelName;
        agentName = SimulationConfiguration.AgentName;
        sessionName = SimulationConfiguration.SessionName;
        ifcModelDirectory = Application.dataPath + "/IFC Models/";


        Button QuitButton = Quit.GetComponent<Button>();
        QuitButton.onClick.AddListener(QuitButtonStart);

        Button exportButton = Export.GetComponent<Button>();
        exportButton.onClick.AddListener(ExportModelClicked);

        Button AddPropertyButton = AddProperty.GetComponent<Button>();
        AddPropertyButton.onClick.AddListener(AddPropertyButtonClicked);

        Button doneParameterButton = doneParameter.GetComponent<Button>();
        doneParameterButton.onClick.AddListener(DoneParamClicked);


        Button DeleteObjectsButton = DeleteObjects.GetComponent<Button>();
        DeleteObjectsButton.onClick.AddListener(DeleteObjectClicked);

        // Get information if the simulator must run
        if (SimulationConfiguration.RunSimulation)
        {
            MainCanvas.gameObject.SetActive(false);
            AddParamCanvas.gameObject.SetActive(false);

            //agentTypeNames = SimulationConfiguration.AgentTypeNames;
            //agentTaskFileNames = SimulationConfiguration.AgentTaskFileNames;
            simulationLengthInMinutes = SimulationConfiguration.SimulationLengthInMinutes;

            agentTypeAndIDDictionary = new Dictionary<string, int_t>();
            distinctAgentTypeNames = agentTypeNames.Distinct().ToList();
            agentArray = new List<Agent>();

            // SHADAN: the simulator simulates 24 hours in the given simulationLengthInMinutes:
            timeScale = simulationLengthInMinutes / (5 * 60);
            endSimulationTime = Time.time + simulationLengthInMinutes * 60;
            GetAgentNameIDsIntoDictionary();
        }

        try
        {
            // Christoph's stuff (Too lazy to figure out what it does)
            PopulateObjectSelector();
            ruleInstanceHighightedObjects = new List<IfcCollectiveObject>();

            // Used to create the Smart Condo
            RenderModel();
            CategorizeGameObjects();
            SetCamera();

            /*
             * This determines if a simulation should be run
            */
            if (SimulationConfiguration.RunSimulation)
            {
                // Creates a Task Planner here because GetAllIfcCollectiveObjects won't work before RenderModel() 
                // This is due to goDictionary not being instantiated before then
                taskPlanner = new TaskPlanner(Time.time, timeScale);
                taskPlanner.ifcObjectList = GetAllIfcCollectiveObjects();

                // Bounding boxes are annoying to look at
                cubesGameObject.SetActive(false);

                // Creates NavMeshes for each distinct Agent Type
                for (int j = 0; j < distinctAgentTypeNames.Count; j++)
                {
                    AddAndBakeNavMeshSurfaces(agentTypeNames[j]);
                }

                // Creates Agents for the number of agents indicated
                for (int i = 0; i < agentTypeNames.Count; i++)
                {
                    // Works only for single agents
                    CreateAndSpawnAgent(agentTypeNames[i], agentTypeAndIDDictionary[agentTypeNames[i]], i);
                }
            }            
        }
        catch (Exception ex)
        {
            Debug.Log(ex.ToString());
        }
    }

    // Spawns an agent (at a specific point in time)
    // Also reads in a text file defining the agent's actions
    public void CreateAndSpawnAgent(string agentName, int agentTypeID, int i)
    {
        GameObject gameObjectAgent = CreateAgentObject(agentName, PrimitiveType.Cylinder, (float)0.5, (float)1.5, (float)0.3, (float)0.5, (float)0.3);

        NavMeshAgent navMeshAgent = gameObjectAgent.AddComponent<NavMeshAgent>();
        navMeshAgent.agentTypeID = agentTypeID;
        navMeshAgent.speed = 2 / timeScale; // = new Vector3(0.1f, 0.1f, 0.1f);
        navMeshAgent.acceleration = 60 / timeScale;
        Agent agent = gameObjectAgent.AddComponent<Agent>();
        //agent.file = agentTaskFileNames[i];
        agent.file = SimulationConfiguration.SessionName + ".JSON";
        agent.timeScale = timeScale;
        agent.taskPlanner = taskPlanner;
        agentArray.Add(agent);

        agent.ReadTextFile();
    }

    // Creates an Agent GameObject in the scene
    private GameObject CreateAgentObject(string name, PrimitiveType primitiveType, float positionY, float positionZ, float scaleX, float scaleY, float scaleZ)
    {
        GameObject gameObjectAgent = GameObject.CreatePrimitive(primitiveType);
        gameObjectAgent.name = name;
        gameObjectAgent.transform.position = new Vector3(0, positionY, positionZ);
        gameObjectAgent.transform.localScale = new Vector3(scaleX, scaleY, scaleZ);

        return gameObjectAgent;
    }


    // Put agent name and IDs into a dictionary
    // The user selects which agent type they would like to use
    private void GetAgentNameIDsIntoDictionary()
    {
        var count = NavMesh.GetSettingsCount();
        var agentTypeNames = new string[count + 2];
        for (var i = 0; i < count; i++)
        {
            var id = NavMesh.GetSettingsByIndex(i).agentTypeID;
            var name = NavMesh.GetSettingsNameFromID(id);

            agentTypeAndIDDictionary.Add(name, id);
        }
    }

    // Add a NavMeshSurface and use the AgentTypeID specified
    // Build a NavMesh for the AgentTypeID
    private void AddAndBakeNavMeshSurfaces(string agentTypeName)
    {
        // TODO: Add a for loop iterating through number of agents and agent types 
        // Something like that, each type of agent needs a different NavMesh generated
        rootGameObject.AddComponent<NavMeshSurface>();
        rootGameObject.GetComponent<NavMeshSurface>().agentTypeID = agentTypeAndIDDictionary[agentTypeName];
        rootGameObject.GetComponent<NavMeshSurface>().overrideVoxelSize = true;
        rootGameObject.GetComponent<NavMeshSurface>().voxelSize = 0.02f;
        rootGameObject.GetComponent<NavMeshSurface>().BuildNavMesh();

        // Change the gameobjects into Navigation Static
        // This is so the NavMesh recognizes these and the NavMeshAgent will work properly
        foreach (Transform child in rootGameObject.transform)
        {
            foreach (Transform grandchild in child.transform)
            {
                GameObjectUtility.SetStaticEditorFlags(grandchild.gameObject, StaticEditorFlags.NavigationStatic);
                GameObjectUtility.SetStaticEditorFlags(child.gameObject, StaticEditorFlags.NavigationStatic);
                GameObjectUtility.SetStaticEditorFlags(rootGameObject, StaticEditorFlags.NavigationStatic);
            }
        }
    }

    private void PopulateObjectSelector()
    {
        // Get all image-object pairs:
        List<GameObject> objects = Resources.LoadAll<GameObject>("Objects").ToList();
        Texture2D[] images2d = Resources.LoadAll<Texture2D>("Images");
        objectTuples = new List<Tuple<GameObject, Sprite>>();
        foreach (GameObject go in objects)
        {
            foreach (Texture2D tex in images2d)
            {
                if (go.name == tex.name)
                {
                    Sprite NewSprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(0, 0), 100.0f);
                    objectTuples.Add(new Tuple<GameObject, Sprite>(go, NewSprite));
                    break;
                }
            }
        }

        int rowCount = (int)Mathf.Ceil(objectTuples.Count / 3.0f);
        ObjectSelectionContent.GetComponent<RectTransform>().sizeDelta = new Vector2(0, rowCount * 100.0f);
        float y0 = (rowCount - 2.0f) / 2.0f * 100.0f;
        for (int i = 0; i < objectTuples.Count; i++)
        {
            Tuple<GameObject, Sprite> tup = objectTuples[i];

            // Create Image Button
            GameObject newObj = new GameObject(tup.Item1.name); //Create the GameObject
            Image newImage = newObj.AddComponent<Image>(); //Add the Image Component script
            Button newButton = newObj.AddComponent<Button>(); //Add the Button Component script
            int index = i;
            newButton.onClick.AddListener(delegate { AddNewObjectImageClicked(index); });
            newImage.sprite = tup.Item2; //Set the Sprite of the Image Component on the new GameObject
            newObj.GetComponent<RectTransform>().SetParent(ObjectSelectionContent.transform); //Assign the newly created Image GameObject as a Child of the Parent Panel.
            newObj.SetActive(true); //Activate the GameObject

            // Modify Content space to fit
            int row = (int)Mathf.Floor(i / 3);
            int col = i % 3;
            RectTransform rectT = newObj.GetComponent<RectTransform>();
            rectT.pivot = new Vector2(1.5f, 0);
            rectT.anchoredPosition = new Vector2(col * 100.0f, y0 + row * -100.0f);
        }
    }

    private int CountCompletedAgents()
    {
        int completedAgents = 0;

        foreach(Agent agent in agentArray)
            if (agent.GetCurrentAction().ToLower() == "finish session")
                completedAgents++;

        return completedAgents;
    }

    // Update is called once per frame
    void Update()
    {
        if (SimulationConfiguration.RunSimulation)
        {
            // If the simulation time is over, end the application
            if (Time.time > endSimulationTime + agentArray[0].getIdleTime() || CountCompletedAgents() == agentArray.Count())
            {
                

                string dir = "Assets\\Logs\\" + DateTime.Now.ToString("yyyy-MM-dd_hh-mm-ss")+"\\";
                Directory.CreateDirectory(dir);
                foreach (Agent agent in agentArray)
                {
                    agent.WriteLog(dir);
                }

                // Use this only while in the Editor
                UnityEditor.EditorApplication.isPlaying = false;

                // Use this if you are actually running the application and not in the Unity Editor
                //Application.Quit();
            }
        }

        try
        {
            MoveCamera();

            // If we are not locking an object or if an object is being added
            if (!objectLocked || objectBeingAdded)
            {
                ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out hit))
                {
                    // We are hovering over an item
                    IfcCollectiveObject objectMouseIsOver = GetAllIfcCollectiveObjects().Find(obj => obj.GameObjectBB == hit.collider.gameObject);
                    if (!objectLocked)
                    {
                        // Check if a new object is being highlighted
                        if (selectedObject != objectMouseIsOver)
                        {
                            if (selectedObject != null)
                            {
                                selectedObject.Unhighlightobjects();
                            }
                            selectedObject = objectMouseIsOver;

                            selectedObject.Highlightobjects(highlightMatYellow);
                            ObjectInfo.text = selectedObject.GetInfoText();
                        }
                    }

                    if (objectBeingAdded)
                    {
                        // If we are adding a new object then check if it a floor "slab" so that we can place the new object on it at that locations
                        if (objectMouseIsOver.IfcType.Contains("Slab"))
                        {
                            Vector3 potentialNewLocation = new Vector3(hit.point.x, hit.point.y, hit.point.z);
                            gameObjBeingAdded.transform.position = potentialNewLocation;
                        }

                        // TODO: May be better to have a box that takes in the height the object will be placed at rather then restrict it to being on the floor
                        // A "work plane" object with a collider attached my do this well were its default is set to the floor slab height. that way the camera
                        // does not need to be fied to the top down camera (maybe even have dual cameras like in 3Ds-Max)
                    }
                }
                else
                {
                    if (selectedObject != null)
                    {   
                        selectedObject.Unhighlightobjects();
                        selectedObject = null;
                    }
                }
            }

            if (Input.GetMouseButtonDown(1))
            {
                objectLocked = (selectedObject != null) ? !objectLocked : false;
                if (objectBeingAdded)
                {
                    PlaceNewObjectAndCheck();
                }

                foreach (IfcCollectiveObject colObj in ruleInstanceHighightedObjects)
                {
                    colObj.Unhighlightobjects();
                }
                ruleInstanceHighightedObjects = new List<IfcCollectiveObject>();
            }
        }
        catch (Exception ex)
        {
            Debug.Log(ex.ToString());
        }
    }

    private void PlaceNewObjectAndCheck()
    {
        Debug.Log(modelScale);
        IfcCollectiveObject newObject = new IfcCollectiveObject(gameObjBeingAdded, IfcDataBase, modelScale);
        AddGameObjectToCategory(newObject);
        CategorizeGameObjects();

        gameObjBeingAdded = null;
        objectBeingAdded = false;
    }

    private void MoveCamera()
    {
        Vector3 camPossition = mainCamera.transform.position;
        float moveAmount = 1.0f / 10.0f;
        if (Input.GetKey(KeyCode.UpArrow))
            mainCamera.transform.position = new Vector3(camPossition.x, camPossition.y, camPossition.z + moveAmount);
        if (Input.GetKey(KeyCode.DownArrow))
            mainCamera.transform.position = new Vector3(camPossition.x, camPossition.y, camPossition.z - moveAmount);
        if (Input.GetKey(KeyCode.LeftArrow))
            mainCamera.transform.position = new Vector3(camPossition.x - moveAmount, camPossition.y, camPossition.z);
        if (Input.GetKey(KeyCode.RightArrow))
            mainCamera.transform.position = new Vector3(camPossition.x + moveAmount, camPossition.y, camPossition.z);
    }

    private void SetCamera()
    {
        // Set the Camera:
        Vector3 center, diment;
        IfcMeshUtils.GetModelCenterAndDims(cubesGameObject, out center, out diment);
        diment = (diment.x <= 0 || diment.y <= 0 || diment.z <= 0) ? new Vector3(100, 100, 100) : diment;

        mainCamera.transform.position = new Vector3(center.x, center.y + diment.y / 2.0f, center.z);
        mainCamera.orthographic = true;
        mainCamera.orthographicSize = Mathf.Max(diment.x / 2.0f, diment.z / 2.0f);
        cameraSize = mainCamera.orthographicSize;
        mainCamera.nearClipPlane = 0;
        mainCamera.farClipPlane = diment.y;
        cameraClipPlane = diment.y;
        mainCamera.transform.eulerAngles = new Vector3(90, 0, 0);
    }

    private void CategorizeGameObjects()
    {
        // Categorize objects
        foreach (KeyValuePair<string, List<IfcCollectiveObject>> kvp in goDictionary)
        {
            GameObject categoryGameObject = GameObject.Find(modelName + ": " + kvp.Key) ?? new GameObject(modelName + ": " + kvp.Key);
            categoryGameObject.transform.parent = rootGameObject.transform;

            GameObject categoryGameObjectBB = GameObject.Find(modelName + ": BB: " + kvp.Key) ?? new GameObject(modelName + ": BB: " + kvp.Key);
            categoryGameObjectBB.transform.parent = cubesGameObject.transform;

            foreach (IfcCollectiveObject ico in kvp.Value)
            {
                if (ico.GameObjectBB != null)
                    ico.GameObjectBB.transform.parent = categoryGameObjectBB.transform;
                if (ico.IfcGameObject != null)
                    ico.IfcGameObject.transform.parent = categoryGameObject.transform;
            }
        }
    }

    private void AddGameObjectToCategory(IfcCollectiveObject newObject)
    {
        // Categorize objects method1:
        string newObjKey = goDictionary.Keys.FirstOrDefault(k => k == newObject.IfcType);
        if (newObjKey != null)
        {
            goDictionary[newObjKey].Add(newObject);
        }
        else
        {
            goDictionary.Add(newObject.IfcType ?? "Null", new List<IfcCollectiveObject>() { newObject });
        }

        // Categorize objects method2:
        //bool foundInDic = false;
        //foreach (KeyValuePair<string, List<IfcCollectiveObject>> kvp in goDictionary)
        //{
        //    if (newObject.IfcType == kvp.Key)
        //    {
        //        kvp.Value.Add(newObject);
        //        foundInDic = true;
        //        break;
        //    }
        //}
        //if (!foundInDic)
        //{
        //    goDictionary.Add(newObject.IfcType ?? "Null", new List<IfcCollectiveObject>() { newObject });
        //}
    }

    private void RenderModel()
    {
        try
        {
            // make sure the modelName is set!
            string file = ifcModelDirectory + modelName + ".ifc";
            IfcDataBase = new DatabaseIfc(file);

            // Shadan:
            //IfcDataBase.ScaleSI = 0.8;

            modelScale = (float)IfcDataBase.ScaleSI;

            rootGameObject = new GameObject(modelName);
            cubesGameObject = new GameObject(modelName + ": BB");
            goDictionary = new Dictionary<string, List<IfcCollectiveObject>>();

            List<IfcElement> elemList = IfcDataBase.Project.Extract<IfcElement>();
            
            foreach (IfcElement elem in elemList)
            {
                try
                {
                    // Create Object:
                    
                    IfcCollectiveObject ifcObjectElem = new IfcCollectiveObject(elem, modelScale);
                    SetMaterial(ifcObjectElem);
                    AddGameObjectToCategory(ifcObjectElem);
                }
                catch (Exception ex)
                {
                    Debug.Log("ERROR: " + elem.Name + "\n" + ex.ToString());
                }
            }
        }
        catch (Exception ex)
        {
            Debug.Log("ERROR: " + ex.ToString());
        }
    }

    public void SetMaterial(IfcCollectiveObject ifcObject)
    {
        //IfcMaterialSelect materialElem = ifcObject.IfcElem.MaterialSelect;
        //if (materialElem != null)
        //{
        //    //Debug.Log(elem.MaterialSelect.Name);
        //    ifcObject.SetMaterial(defaultMat4);
        //}

        List<Material> matList = Resources.LoadAll<Material>("Materials").ToList();
        string objectEverythingString = ifcObject.GetLongStringThatMayContainMaterial().ToLower();
        bool matFound = false;
        foreach (Material mat in matList)
        {
            if (objectEverythingString.Contains(mat.name.ToLower()))
            {
                matFound = true;
                ifcObject.SetMaterial(mat);
                break;
            }
        }

        if (!matFound)
        {
            // If Material is being set by object:
            if (ifcObject.IfcType.Contains("Wall"))
            {
                ifcObject.SetMaterial(defaultMat4);
                return;
            }
            if (ifcObject.IfcType.Contains("Furnishing"))
            {
                ifcObject.SetMaterial(defaultMat3);
                return;
            }
            if (ifcObject.IfcType.Contains("Door") || ifcObject.IfcType.Contains("Window"))
            {
                ifcObject.SetMaterial(defaultMat1);
                return;
            }
            ifcObject.SetMaterial(defaultMat2);
        }
    }

    private void DeactivateAllGameObjectCameras(GameObject go)
    {
        for (int i = 0; i < go.transform.childCount; i++)
        {
            GameObject child = go.transform.GetChild(i).gameObject;
            DeactivateAllGameObjectCameras(child);
            if (child.GetComponent<Camera>() != null)
            {
                child.SetActive(false);
            }
        }
    }

    public List<IfcCollectiveObject> GetAllIfcCollectiveObjects()
    {
        List<IfcCollectiveObject> allObjects = goDictionary.Values.SelectMany(x => x).ToList();
        return allObjects;
    }

    public void DestroyGameObjectAndChildren(GameObject go)
    {
        for (int i = 0; i < go.transform.childCount; i++)
        {
            DestroyGameObjectAndChildren(go.transform.GetChild(i).gameObject);
        }
        Destroy(go);
    }

    public void DestroyGameObjectChildren(GameObject go)
    {
        for (int i = 0; i < go.transform.childCount; i++)
        {
            Destroy(go.transform.GetChild(i).gameObject);
        }
    }

    // Interface Events:

    public void CheckModelClicked()
    {
        List<IfcCollectiveObject> objects = GetAllIfcCollectiveObjects();
        for (int i = 0; i < objects.Count - 1; i++)
        {
            IfcCollectiveObject ifcco1 = objects[i];
            // Only check the next objects not the ones already done:
            List<IfcCollectiveObject> checkObjects = objects.GetRange(i, objects.Count - i);
        }
    }

    public void ZoomCamera()
    {
        mainCamera.orthographicSize = ZoomSlider.value * 2 * cameraSize;
    }

    public void DeleteObjectClicked()
    {
        try
        {    
            if (objectLocked && selectedObject != null)
            {
                selectedObject.DeactivateObjects();
                //selectedObject.Properties.Remove(selectedObject.Properties.Last());
                selectedObject = null;
                objectLocked = false;
                ObjectInfo.text = "Object Info";
	        
            }
        }
        catch (Exception ex)
        {
            Debug.Log(ex.ToString());
        }
    }

    public void AddNewObjectImageClicked(int goIndex)
    {
        try
        {
            if (objectLocked && selectedObject != null)
            {
                selectedObject = null;
                objectLocked = false;
                ObjectInfo.text = "Object Info";
            }

            GameObject newObj = objectTuples[goIndex].Item1;
            gameObjBeingAdded = Instantiate(newObj);
            DeactivateAllGameObjectCameras(gameObjBeingAdded);

            Vector3 objectScale = gameObjBeingAdded.transform.localScale;
            float scaleAmount = INCH_TO_METER;
            gameObjBeingAdded.transform.localScale = new Vector3(objectScale.x * scaleAmount, objectScale.y * scaleAmount, objectScale.z * scaleAmount);

            Vector3 center, diment;
            IfcMeshUtils.GetModelCenterAndDims(cubesGameObject, out center, out diment);
            gameObjBeingAdded.transform.position = center;

            objectBeingAdded = true;
        }
        catch (Exception ex)
        {
            Debug.Log(ex.ToString());
        }
    }

    public void ExportModelClicked()
    {
        string newFileName = ifcModelDirectory + ExportFileName.text;
        IfcDataBase.WriteFile(newFileName + ".ifc");


    }

    public void ToggleChangedObjects()
    {
        List<IfcCollectiveObject> allObjects = GetAllIfcCollectiveObjects();
        foreach (IfcCollectiveObject ico in allObjects)
        {
            ico.ToggleMeshRenderer(ToggleObjects.GetComponent<Toggle>().isOn);
        }
    }

    public void SliderTopValueChanged()
    {
        TopSlider.value = Mathf.Min(TopSlider.value, (1.0f - 0.1f));
        if (TopSlider.value + 0.1f > BottomSlider.value)
        {
            BottomSlider.value = TopSlider.value + 0.1f;
        }
        mainCamera.nearClipPlane = TopSlider.value * cameraClipPlane;
        mainCamera.farClipPlane = BottomSlider.value * cameraClipPlane;
    }

    public void SliderBottomValueChanged()
    {
        BottomSlider.value = Mathf.Max(BottomSlider.value, 0.1f);
        if (TopSlider.value > BottomSlider.value - 0.1f)
        {
            TopSlider.value = BottomSlider.value - 0.1f;
        }
        mainCamera.nearClipPlane = TopSlider.value * cameraClipPlane;
        mainCamera.farClipPlane = BottomSlider.value * cameraClipPlane;
    }

    public void AddPropertyButtonClicked()
    {
        // Open a Canvas that will have a textbox for both the property name and the Value.
        if (objectLocked)
        {
            AddParamCanvas.gameObject.SetActive(true);
        }
    }

    public void DoneParamClicked()
    {
        // Should trigger the new object to add the new property in the IFC file aswell (under property set)
        string parameterName = ParamNameText.text;
        string parameterValue = ParamValueText.text;

        selectedObject.AddElementProperty(selectedObject.IfcElem, IfcDataBase, parameterName, parameterValue);
        AddParamCanvas.gameObject.SetActive(false);
    }

    public void CancelParamClickec()
    {
        AddParamCanvas.gameObject.SetActive(false);
    }

    public void QuitButtonStart()
    {
        // Use this only while in the Editor
        UnityEditor.EditorApplication.isPlaying = false;

        // Use this if you are actually running the application and not in the Unity Editor
        //Application.Quit();
    }
}