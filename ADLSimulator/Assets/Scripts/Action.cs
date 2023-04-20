using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SimpleJSON;
using System.Linq;

public class Action {

    private List<string> aliases;
    private List<string> dependencies;
    private List<string> postActions;
    private List<int> times;

    public string name;
    public float probability;
    public float duration;
    public int occurrence;
    public int completed;

    public Action (int id, int duration, int priority = 100, List<string> times = null, List<string> aliases = null, List<string> dependencies = null){

        //TODO should have another way to initialize

    }

    public Action(JSONNode action)
    {
        //may need to do further parsing
        aliases = new List<string>();
        dependencies = new List<string>();
        postActions = new List<string>();
        times = new List<int>();

        name = action["name"];
        probability = action["probability"].AsFloat;
        duration = action["duration"].AsFloat;

        occurrence = 0;
        completed = 0;
        
        var keyList = action.Keys;
        List<string> list = new List<string>();
        foreach (var key in keyList) {
            list.Add(key);
        }
        
        if (list.Contains("occurrence"))
        {
            occurrence = action["occurrence"].AsInt;
        }

        for (int i = 0; i < action["requires"].Count; i++) {
            dependencies.Add(action["requires"][i]);
        }

        for (int i = 0; i < action["post"].Count; i++)
        {
            postActions.Add(action["post"][i]);
        }

        for (int i = 0; i < action["aliases"].Count; i++)
        {
            aliases.Add(action["aliases"][i]);
        }

        for (int i = 0; i < action["times"].Count; i++)
        {
            times.Add(action["times"][i][0].AsInt);
            times.Add(action["times"][i][1].AsInt);
        }
    }

    public string GetAlias() {
        System.Random random = new System.Random();
        if (aliases.Count == 0) {
            return name;
        }
        return aliases[random.Next(aliases.Count)];
    }

    public string GetDependency()
    {
        // if (name == "Eat") {
        //     Debug.Log("Dependencies:");
        //     foreach (string a in dependencies) {
        //         Debug.Log(a);
        //     }
        // }
        System.Random random = new System.Random();
        if (dependencies.Count > 0)
        {
            return dependencies[random.Next(dependencies.Count)];
        }
        // Debug.Log("Action found no dependency");
        return null;
    }

    public string GetPostAction()
    {
        System.Random random = new System.Random();
        if (postActions.Count > 0)
        {
            return postActions[random.Next(postActions.Count)];
        }
        return null;
    }

    public bool availableAtTime(float time, float timeScale)
    {
        if (times.Count == 0)
        {
            return true;
        }
        if (name == "Eat")
        {
            Debug.Log("Eat time: " + time);
            Debug.Log("Eat min: " + times[0] * 3600 * timeScale);
            Debug.Log("Eat max: " + times[1] * 3600 * timeScale);
            Debug.Log("Eat time check: " + (time >= times[0] * 3600 * timeScale && time <= times[1] * 3600 * timeScale));
        }
        for (int i = 0; i < times.Count/2; i++) {
            if (time >= times[2 * i] * 3600 * timeScale && time <= times[2 * i + 1] * 3600 * timeScale) {
                return true;
            }
        }

        return false;
    }

    public override string ToString()
    {
        return "Action: { Name:" + name + 
                        " probability: "   + probability + 
                        " duration: "   + duration + 
                        " required: "   + dependencies.ToString() +
                        " postActions: " + postActions.ToString() +
                        " aliases "     + aliases.ToString() +
                        " times "       + times.ToString() +            
                        "}";
    }
}
