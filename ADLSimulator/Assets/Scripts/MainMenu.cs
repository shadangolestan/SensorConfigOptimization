using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class MainMenu : MonoBehaviour
{

    // Main Menu UI
    public Button BIMEditor;
    public Button ADLSimulation;
    public Button Quit;

    // Model Select UI
    public Dropdown ModelSelectDropdown;
    public Dropdown AgentSelectDropdown;
    public Button ModelSelectContinue;
    public Button ModelSelectBack;
    public Text ModelSelectionError;

    // Simulation Length UI
    public InputField simulationLengthInputField;
    public Button SimulationLengthContinue;
    public Button SimulationLengthBack;
    public Text SimulationLengthError;

    public GameObject mainMenuCanvas;
    public GameObject modelSelectCanvas;
    public GameObject simulationLengthCanvas;

    private string ifcModelDirectory;
    private string agentDirectory;

    void Start()
    {

        ifcModelDirectory = Application.dataPath + "/IFC Models/";
        agentDirectory = Application.dataPath + "/Task Files/";

        SetupModelSelectDropDown();
        SetupAgentSelectDropDown();
        SetupSimulationLengthInput();

        Button BIMEditorButton = BIMEditor.GetComponent<Button>();
        BIMEditorButton.onClick.AddListener(BIMEditorStart);

        Button ADLSimulationButton = ADLSimulation.GetComponent<Button>();
        ADLSimulationButton.onClick.AddListener(ADLSimulationStart);

        Button QuitButton = Quit.GetComponent<Button>();
        QuitButton.onClick.AddListener(QuitButtonStart);

        Button ModelSelectContinueButton = ModelSelectContinue.GetComponent<Button>();
        ModelSelectContinueButton.onClick.AddListener(ModelSelectContinueStart);

        Button ModelSelectBackButton = ModelSelectBack.GetComponent<Button>();
        ModelSelectBackButton.onClick.AddListener(ModelSelectBackStart);

        Button SimulationLengthContinueButton = SimulationLengthContinue.GetComponent<Button>();
        SimulationLengthContinueButton.onClick.AddListener(SimulationLengthContinueStart);

        Button SimulationLengthBackButton = SimulationLengthBack.GetComponent<Button>();
        SimulationLengthBackButton.onClick.AddListener(SimulationLengthBackStart);

    }

    private void BIMEditorStart()
    {
        SimulationConfiguration.RunSimulation = false;
        SwapToModelSelectMenu();
    }

    private void ADLSimulationStart()
    {
        SimulationConfiguration.RunSimulation = true;
        SwapToModelSelectMenu();
    }

    private void QuitButtonStart()
    {
        Application.Quit();
    }

    private void ModelSelectBackStart()
    {
        modelSelectCanvas.SetActive(false);
        mainMenuCanvas.SetActive(true);
    }

    // Depending on if user chose editor or simulator, give different results
    private void ModelSelectContinueStart()
    {
        if (SimulationConfiguration.RunSimulation)
        {
            modelSelectCanvas.SetActive(false);
            simulationLengthCanvas.SetActive(true);
        }
        else
        {
            SceneManager.LoadScene("Simulator Scene");
        }
    }

    private void SimulationLengthContinueStart()
    {
        SceneManager.LoadScene("Simulator Scene");
    }

    private void SimulationLengthBackStart()
    {
        simulationLengthCanvas.SetActive(false);
        modelSelectCanvas.SetActive(true);
    }

    private void SwapToModelSelectMenu()
    {
        mainMenuCanvas.SetActive(false);
        modelSelectCanvas.SetActive(true);
    }

    // Sets up the available objects for the dropdown
    private void SetupModelSelectDropDown()
    {
        DirectoryInfo dir = new DirectoryInfo(ifcModelDirectory);
        FileInfo[] fileInfos = dir.GetFiles("*.*");
        List<Dropdown.OptionData> options = new List<Dropdown.OptionData>();

        // Add a blank dropdown option and make sure the user picks an option that is not blank
        options.Add(new Dropdown.OptionData() { text = "" });
        ModelSelectContinue.interactable = false;

        foreach (FileInfo fi in fileInfos)
        {
            if (fi.Extension != ".meta")
            {
                options.Add(new Dropdown.OptionData(Path.GetFileNameWithoutExtension(fi.Name)));
            }
        }

        ModelSelectDropdown.ClearOptions();
        ModelSelectDropdown.AddOptions(options);

        ModelSelectDropdown.onValueChanged.AddListener(delegate { ModelSelectDropdownValueChanged(); });
    }

    private void SetupAgentSelectDropDown()
    {
        DirectoryInfo dir = new DirectoryInfo(agentDirectory);
        FileInfo[] fileInfos = dir.GetFiles("*.*");
        List<Dropdown.OptionData> options = new List<Dropdown.OptionData>();

        // Add a blank dropdown option and make sure the user picks an option that is not blank
        options.Add(new Dropdown.OptionData() { text = "" });
        ModelSelectContinue.interactable = false;

        foreach (FileInfo fi in fileInfos)
        {
            if (fi.Extension != ".meta")
            {
                options.Add(new Dropdown.OptionData(Path.GetFileNameWithoutExtension(fi.Name)));
            }
        }

        AgentSelectDropdown.ClearOptions();
        AgentSelectDropdown.AddOptions(options);

        AgentSelectDropdown.onValueChanged.AddListener(delegate { AgentSelectDropdownValueChanged(); });
    }


    // Makes the continue button unclickable and displays error message if it's blank
    // Otherwise, allows user to continue when given a valid number
    private void ModelSelectDropdownValueChanged()
    {
        if (ModelSelectDropdown.value == 0)
        {
            ModelSelectionError.gameObject.SetActive(true);
            ModelSelectContinue.interactable = false;
        }
        else
        {
            ModelSelectionError.gameObject.SetActive(false);
            ModelSelectContinue.interactable = true;
            SimulationConfiguration.ModelName = ModelSelectDropdown.options[ModelSelectDropdown.value].text;
        }
    }

    private void AgentSelectDropdownValueChanged()
    {
        if (AgentSelectDropdown.value == 0)
        {
            ModelSelectionError.gameObject.SetActive(true);
            ModelSelectContinue.interactable = false;
        }
        else
        {
            ModelSelectionError.gameObject.SetActive(false);
            ModelSelectContinue.interactable = true;
            SimulationConfiguration.SessionName = AgentSelectDropdown.options[AgentSelectDropdown.value].text;
            SimulationConfiguration.AgentName = "Agent1";
        }
    }

    // Sets up the simulation length input field for error checking
    private void SetupSimulationLengthInput()
    {
        simulationLengthInputField.onValueChanged.AddListener(delegate { SimulationInputValueChanged(); });
    }

    private void SimulationInputValueChanged()
    {
        if (simulationLengthInputField.text == "" || float.Parse(simulationLengthInputField.text) <= 0)
        {
            SimulationLengthError.gameObject.SetActive(true);
            SimulationLengthContinue.interactable = false;
        }
        else
        {
            SimulationLengthError.gameObject.SetActive(false);
            SimulationLengthContinue.interactable = true;
            SimulationConfiguration.SimulationLengthInMinutes = float.Parse(simulationLengthInputField.text);
        }
    }

}
