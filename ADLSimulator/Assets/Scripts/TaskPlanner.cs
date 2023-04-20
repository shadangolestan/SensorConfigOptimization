using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.AI;
using System.Linq;

public class TaskPlanner {

    private List<GameObject> actionableIFCGameObjects;
    private List<Vector3> actionableDestinations;
    public List<IfcCollectiveObject> ifcObjectList { get; set; }

    private float timeScale;
    private float startTime;

    private System.Random random;

    // Use this for initialization
   public TaskPlanner(float time, float timemod)
    {
        timeScale = timemod;
        startTime = time;
        actionableIFCGameObjects = new List<GameObject>();
        actionableDestinations = new List<Vector3>();
        random = new System.Random();
    }

    // TODO: Do a time calculation for actions to make sure there's enough time to complete them
    // Randomly selects a viable action for the agent
    public string GetNextAgentAction(Dictionary<string, Action> dictionaryActions, List<string> completedActions)
    {
        // Shuffles the actions around
        List<Action> dActions = dictionaryActions.Values.ToList();
        System.Random random = new System.Random();
        List<Action> actions = dActions; // dActions.OrderBy(x => random.Next()).ToList();

        List<Action> guarenteedActions = new List<Action>();
        for (int i = 0; i < actions.Count; i++)
        {
            if (actions[i].probability == 100)
            {
                guarenteedActions.Add(actions[i]);
                actions.RemoveAt(i);
                i--;
            }
        }

        // Debug.Log(dActions);
        // Debug.Log(actions);
        Action action;

        string actionName = null;
        try
        {
            action = actions[0];
        }
        catch
        {
            action = guarenteedActions[0];
        }

        // Checks for postcondition
        if (completedActions.Count != 0)
        {
            action = dictionaryActions[completedActions.Last()];

            actionName = action.GetPostAction();
            if (actionName != null)
            {
                action = dictionaryActions[actionName];
                action.completed++;
                completedActions.Add(actionName);

                return actionName;
            }
        }

        // Probability
        // Debug.Log("Checking Guaranteed actions");
        // Debug.Log(guarenteedActions.Count);
        // Debug.Log(guarenteedActions[0].name);
        // Debug.Log(guarenteedActions[1].name);
        actionName = findAction(guarenteedActions, completedActions);

        if (actionName == null)
        {
            // Debug.Log("Checking all actions");
            // Debug.Log(actions.Count);
            actionName = findAction(actions, completedActions);
        }

        try
        {
            action = dictionaryActions[actionName];

            string dependencyName = action.GetDependency();
            if (dependencyName != null)
            {
                Debug.Log("Dependency Found: " + actionName);
                action = dictionaryActions[dependencyName];
                //!action.completed++;
                //!completedActions.Add(dependencyName);
                actionName = dependencyName;
            }
            Debug.Log("There is no dependency");

            action.completed++;
            Debug.Log(actionName);
            completedActions.Add(actionName);

            return actionName;
        }
        catch
        {
            // Goes to catch when actionName == null
            return null;
        }
    }

    private string findAction(List<Action> actions, List<string> completedActions)
    {
        Action action;

        for (int i = 0; i < actions.Count; i++)
        {
            action = actions[i];
            Debug.Log(action.name);

            int randVar = random.Next(101);

            if (action.probability + randVar < 100)
            {
                Debug.Log("Failed chance");
                continue;
            }

            if (action.occurrence > 0 && action.occurrence <= action.completed)
            {
                Debug.Log("Failed max occurrences");
                continue;
            }

            if (!action.availableAtTime(Time.time - startTime, timeScale))
            {
                Debug.Log("Failed time");
                continue;
            }

            if (IsRepetitiveAction(action.name, completedActions, 2))
            {
                Debug.Log("Failed frequency");
                continue;
            }
            Debug.Log("Returning:" + action.name);
            return action.name;
        }
        return null;
    }

    // Checks if the action selected is a repeat 
    public bool IsRepetitiveAction(string actionName, List<string> completedActions, int length = 1)
    {
        // Checks the most recently completed actions, where the amount is determined by length
        // By default, it's set to 1
        for (int i = completedActions.Count; i > (completedActions.Count - Mathf.Min(completedActions.Count, length)); i--)
        {
            if (completedActions[i - 1].Equals(actionName))
            {
                return true;
            }
        }
        return false;
    }


    // We currently recalculate the objects and destinations for each action and there is no form of caching
    // TODO: Prepopulate everything or cache it somehow to prevent unnecessary calculation
    public Vector3 GetActionDestination(string action)
    {
        if (action.Equals("finish session"))
        {
            return new Vector3(0,0,0);
        }

        FindActionableIFCGameObjects(action);
        FindClosestPointsOnNavMesh();

        Vector3 destination = new Vector3(0, 0, 0);
        try
        {
            Debug.Log(action);
            Debug.Log(actionableDestinations.Count);
            destination = actionableDestinations[random.Next(actionableDestinations.Count)];
        }
        catch {
            destination = actionableDestinations[random.Next(actionableDestinations.Count)];
        }


        

        // BoxCollider rect = floor.GetComponent<BoxCollider>();
        // Debug.Log(new Vector3(-rect.bounds.size.x / 2, 0, rect.bounds.size.z / 2));



  
        return destination;
    }

    // Finds the list of objects that have the action value associated with them
    // Stores these objects inside of an array
    private void FindActionableIFCGameObjects(string actionValue)
    {
        actionableIFCGameObjects.Clear();
        
        for (int i = 0; i < ifcObjectList.Count; i++)
        {
            for (int j = 0; j < ifcObjectList[i].Properties.Count; j++)
            {
                // Only matches values under the assumption that no other type of properties will have that value
                List<string> actions = ifcObjectList[i].Properties[j].PropertyValue.ToString().ToLower().Split(',').ToList<string>();

                foreach (string str in actions)
                {
                    if (str.Trim() == actionValue.ToLower())
                    {
                        actionableIFCGameObjects.Add(ifcObjectList[i].IfcGameObject);
                    }
                }
            }
        }
    }

    // Goes through a list of possible actionable objects and finds the closest point to them on the NavMesh
    private void FindClosestPointsOnNavMesh()
    {
        actionableDestinations.Clear();

        for (int i = 0; i < actionableIFCGameObjects.Count; i++)
        {
            NavMeshHit hit;
            Vector3 result;
            Vector3 center = actionableIFCGameObjects[i].GetComponent<Renderer>().bounds.center;

            if (NavMesh.SamplePosition(center, out hit, 2.0f, NavMesh.AllAreas))
            {
                result = hit.position;
                actionableDestinations.Add(result);
            }

        }
    }
}
