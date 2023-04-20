using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.AI;
using SimpleJSON;
using System;
//using System.IO;

public class Agent : MonoBehaviour
{
    public string file;
    private string jsonString;
    private List<string> completedActions;
    private Dictionary<string, Action> availableActions;


    // As this variable name may be misleading, this is how long a hour / minute / second lasts inside the simulation
    public float timeScale;

    // Action details
    public TaskPlanner taskPlanner;
    private float actionEndTime;
    private string currentAction;
    private string actionAlias;
    private float agentStart;
    private List<string> log;
    private NavMeshAgent navMeshAgent;
    private float walkingTime;
    private bool isWalking;
    private float walkingDuration;
    private float idleTime;
    private float previousDataPointTime;
    private GameObject floor;

    public void Start()
    {
        idleTime = 0f;
        walkingDuration = 0f;
        isWalking = false;
        walkingTime = 0f;
        completedActions = new List<string>();
        log = new List<string>();
        navMeshAgent = gameObject.GetComponent<NavMeshAgent>();
        currentAction = "";
        previousDataPointTime = Time.time;
        actionEndTime = Time.time;
        agentStart = Time.time;

        floor = FindFloor();
    }

    public string GetCurrentAction()
    {
        return currentAction;
    }

    public float getIdleTime()
    {
        return idleTime;
    }

    private GameObject FindFloor()
    {

        GameObject[] gameObjects = (GameObject[])UnityEngine.Object.FindObjectsOfType(typeof(GameObject));

        for (var i = 0; i < gameObjects.Length; i++)
        {
            if (gameObjects[i].name.StartsWith("Floor"))
            {
                gameObjects[i].AddComponent<BoxCollider>();
                return gameObjects[i];
            }
        }

        return null;
    }

    // Checks if the agent has finished performing the action selected
    private void Update()
    {

        //TODO update a string containing text to dump to file
        Vector3 position = gameObject.transform.position;
        if (Time.time - previousDataPointTime >= 2 * timeScale && Time.time - previousDataPointTime <= 3 * timeScale)
        //if (Time.time - previousDataPointTime == 3 * timeScale)
        {
            /*
             *    cent.x = cent.x - 2.34f;
                  cent.x = cent.x * -1f;
                  cent.z = cent.z + 3.5f; 
             * TIME, AGENTID, X, Z, Y, ACTIVITY
             */

            // Debug.LogWarning("Data Point Stored");

            previousDataPointTime = Time.time;

            //log.Add(GetTimeString() + ","
            //    + "1" + "," /* Fake Agent ID */
            //    + -1 * (position.x - 2.2f) + ","
            //    + ((position.z + 3.2f)) + ","
            //    + position.y + ","
            //    + currentAction + '_' + actionAlias + ",");

            
            BoxCollider rect = floor.GetComponent<BoxCollider>();
            //Debug.Log(new Vector3(-rect.bounds.size.x / 2, 0, rect.bounds.size.z / 2));


            log.Add(GetTimeString() + ","
               + "1" + "," /* Fake Agent ID */
               + (position.x + (rect.bounds.size.x / 2) - rect.bounds.center.x) + ","
               + (-position.z + rect.bounds.center.z + (rect.bounds.size.z / 2)) + ","
               + position.y + ","
               + currentAction + '_' + actionAlias + ",");

            //log.Add(GetTimeString() + ","
            //    + "1" + "," /* Fake Agent ID */
            //    + (-1 * (position.x - 2.23f)) + ","
            //    + (position.z + 3.5f) + ","
            //    + position.y + ","
            //    + currentAction + ",");

            //log.Add(GetTimeString() + ","
            //    + (Time.time - agentStart) + ","
            //    + currentAction + ","
            //    + "1" + "," /* Fake Agent ID */
            //    + (-1 * (position.x - 2.23f)) + ","
            //    + position.y + ","
            //    + (position.z + 3.5f) + ","
            //    + actionAlias);
        }

        else if(Time.time - previousDataPointTime > 3.3 * timeScale)
        {
            previousDataPointTime = Time.time;
        }

        Debug.Log(currentAction +" - " + actionAlias + " - " + GetTimeString());

        if (!CheckDestinationReached() && !isWalking)
        {
            walkingTime = Time.time;
            isWalking = true;
        }
        if (CheckDestinationReached() && isWalking)
        {
            walkingDuration = Time.time - walkingTime;
            isWalking = false;
        }

        if (Time.time > actionEndTime + walkingDuration && CheckDestinationReached())
        {
            UpdateActionInformation();
            idleTime += walkingDuration;
            walkingDuration = 0f;
        }
    }

    private void UpdateActionInformation()
    {
        if (currentAction != "")
        {
            completedActions.Add(currentAction);
        }

        currentAction = taskPlanner.GetNextAgentAction(availableActions, completedActions);
        
        if (currentAction == null)
        {
            Debug.Log("END OF SESSION");
            currentAction = "finish session";
            return;
        }

        Debug.Log(currentAction);

        Action action;

        action = availableActions[currentAction];
        // Debug.Log(action.name);
        // Debug.Log(action.duration);
        // Debug.Log(action.probability);

        System.Random rand = new System.Random();
        float duration = action.duration;
        float actionVariance = (duration / 3) + rand.Next((int)(duration / 2), (int)duration);
        Debug.Log("action duration is : " + duration.ToString());
        Debug.Log("action variance is : " + actionVariance.ToString());
        actionEndTime = Time.time + ((actionVariance * timeScale) * 60);

        actionAlias = action.GetAlias();

        // If the agent isn't doing nothing, then move
        if (!currentAction.ToLower().Equals("nothing"))
        {
            MoveToLocation(taskPlanner.GetActionDestination(currentAction));
            
            // These are for testing the scale...
            // MoveToLocation(new Vector3(-1 * (0f - 2.2f), 0, 0f + 3.2f));
            // MoveToLocation(new Vector3(-1 * (5.5f - 2.2f), 0, (5f - 3.2f)));
        }
    }

    public void ReadTextFile()
    {
        string taskFile = Application.dataPath + "/Task Files/" + file;
        StreamReader streamReader = new StreamReader(taskFile);
        jsonString = streamReader.ReadToEnd();
        streamReader.Close();

        // Parse the JSON
        var parsedJSON = JSON.Parse(jsonString);
        availableActions = new Dictionary<string, Action>();
        Action action;

        for (int i = 0; i < parsedJSON["Actions"].Count; i++)
        {
            action = new Action(parsedJSON["Actions"][i]);
            availableActions.Add(action.name, action);
        }
    }

    private bool CheckDestinationReached()
    {
        return navMeshAgent.velocity.sqrMagnitude == 0f;
    }
    // Tells the agent to move to a point on the NavMesh
    private void MoveToLocation(Vector3 targetPoint)
    {
        navMeshAgent.destination = targetPoint;
        navMeshAgent.isStopped = false;
    }

    private string GetTimeString()
    {
        float time = (Time.time - agentStart) / timeScale;

        TimeSpan timeSpan = TimeSpan.FromSeconds(time);
        string timeString = timeSpan.ToString(@"hh\:mm\:ss");

        return timeString;
    }

    public void WriteLog(string directory)
    {
        string path = directory + "AgentTrace_" + SimulationConfiguration.SessionName.Split('_')[0] + ".csv";
        // string header = "Time,Time(s),Action Type,Action Alias,X,Y,Z";
        string header = "Time,Agent,x,y,z,Action";
        System.IO.File.AppendAllText(path, header + Environment.NewLine);
        string output = String.Join(Environment.NewLine, log);
        System.IO.File.AppendAllText(path, output);
    }
}
