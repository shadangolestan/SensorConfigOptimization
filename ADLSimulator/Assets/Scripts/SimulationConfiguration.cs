using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class SimulationConfiguration
{
    public static bool RunSimulation
    {
        get; set;
    }

    public static string ModelName
    {
        get; set;
    }

    public static string AgentName
    {
        get; set;
    }

    public static string SessionName
    {
        get; set;
    }

    public static float SimulationLengthInMinutes
    {
        get; set;
    }

    public static List<string> AgentTypeNames
    {
        get; set;
    }

    public static List<string> AgentTaskFileNames
    {
        get; set;
    }

}
