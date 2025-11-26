using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

[System.Serializable]
public class LayerActivation
{
    public string name;
    public List<float> activations;
}

[System.Serializable]
public class InferenceTrace
{
    public int true_label;
    public int predicted_label;
    public List<LayerActivation> layers;
}

public class ThoughtProcess : MonoBehaviour
{
    [Header("References")]
    public AtlasGenerator atlasGenerator;
    public GameObject mainOrbPrefab;
    public GameObject childOrbPrefab;

    [Header("Files")]
    public string inferenceJsonPath = "inference_trace.json";

    [Header("Animation Settings")]
    public float mainOrbSpeed = 10f;
    public float childOrbSpeed = 30f;
    public float activationThreshold = 0.5f;
    public float minOrbSize = 0.3f;
    public float maxOrbSize = 2f;
    public float layerDelayTime = 1f;

    [Header("Camera Follow")]
    public bool followMainOrb = true;
    public Vector3 cameraOffset = new Vector3(0, 15, -25);
    public float cameraSmoothSpeed = 5f;
    public float mouseSensitivity = 3f;
    public float mouseWheelSensitivity = 10f;

    private InferenceTrace inferenceData;
    private GameObject mainOrb;
    private bool isAnimating = false;
    private Camera mainCamera;
    private float rotationX = 0f;
    private float rotationY = 20f;
    private float cameraDistance = 50f;
    private DrawingInterface drawingInterface;

    void Start()
    {
        LoadInferenceData();
        mainCamera = Camera.main;

        if (mainCamera == null)
        {
            Debug.LogWarning("No main camera found! Camera follow will not work.");
        }

        // Initialize camera rotation from current camera rotation
        if (mainCamera != null)
        {
            Vector3 euler = mainCamera.transform.eulerAngles;
            rotationY = euler.x;
            rotationX = euler.y;
            cameraDistance = Vector3.Distance(mainCamera.transform.position, Vector3.zero);
        }

        // Get DrawingInterface reference
        drawingInterface = FindObjectOfType<DrawingInterface>();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space) && !isAnimating)
        {
            StartCoroutine(AnimateThoughtProcess());
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            StopAllCoroutines();
            ResetVisualization();
        }

        // Camera follow and rotation logic
        if (followMainOrb && mainOrb != null && mainCamera != null)
        {
            // Mouse rotation (right mouse button)
            if (Input.GetMouseButton(1))
            {
                rotationX += Input.GetAxis("Mouse X") * mouseSensitivity;
                rotationY -= Input.GetAxis("Mouse Y") * mouseSensitivity;
                rotationY = Mathf.Clamp(rotationY, -90f, 90f);
            }

            // Mouse wheel zoom
            float scrollInput = Input.GetAxis("Mouse ScrollWheel");
            if (scrollInput != 0f)
            {
                cameraDistance -= scrollInput * mouseWheelSensitivity;
                cameraDistance = Mathf.Clamp(cameraDistance, 5f, 200f);
            }

            // Calculate camera position based on rotation and distance
            Quaternion rotation = Quaternion.Euler(rotationY, rotationX, 0);
            Vector3 offset = rotation * new Vector3(0, 0, -cameraDistance);
            Vector3 desiredPosition = mainOrb.transform.position + offset;

            // Smooth camera movement
            Vector3 smoothedPosition = Vector3.Lerp(mainCamera.transform.position, desiredPosition, cameraSmoothSpeed * Time.deltaTime);
            mainCamera.transform.position = smoothedPosition;
            mainCamera.transform.LookAt(mainOrb.transform);
        }
    }

    void LoadInferenceData()
    {
        string fullPath = Path.Combine(Application.dataPath, inferenceJsonPath);

        if (!File.Exists(fullPath))
        {
            Debug.LogError($"Inference file not found: {fullPath}");
            return;
        }

        string jsonContent = File.ReadAllText(fullPath);
        inferenceData = JsonUtility.FromJson<InferenceTrace>(jsonContent);

        if (inferenceData != null)
        {
            Debug.Log($"Loaded inference trace: True={inferenceData.true_label}, Predicted={inferenceData.predicted_label}");
        }
    }

    // Public method to reload inference data after user drawing
    public void ReloadInferenceData()
    {
        LoadInferenceData();
        Debug.Log("Inference data reloaded. Ready for visualization!");
    }

    IEnumerator AnimateThoughtProcess()
    {
        if (inferenceData == null || atlasGenerator == null)
        {
            Debug.LogError("Missing data or atlas generator!");
            yield break;
        }

        isAnimating = true;

        // Hide the drawing canvas when animation starts
        if (drawingInterface != null)
        {
            drawingInterface.HideCanvas();
        }

        AtlasStructure atlasData = atlasGenerator.GetAtlasData();

        // Create main orb at starting position
        Vector3 startPos = new Vector3(0, 0, -10);
        mainOrb = Instantiate(mainOrbPrefab, startPos, Quaternion.identity);
        ConfigureOrb(mainOrb, 3f);

        // Animate through each layer
        for (int i = 0; i < inferenceData.layers.Count; i++)
        {
            LayerActivation layerAct = inferenceData.layers[i];
            Layer layerStruct = atlasData.layers[i];

            // Move main orb to layer position
            Vector3 layerCenter = new Vector3(0, 0, layerStruct.z_offset);
            yield return StartCoroutine(MoveOrb(mainOrb, layerCenter, mainOrbSpeed));

            // Spawn neurons for this layer when orb reaches it
            if (!atlasGenerator.IsLayerSpawned(layerAct.name))
            {
                atlasGenerator.GenerateLayerNeurons(layerAct.name);
                yield return new WaitForSeconds(0.5f); // Wait for pop-in animation
            }

            // Pause at layer
            yield return new WaitForSeconds(layerDelayTime * 0.5f);

            // Spawn child orbs for activated neurons
            List<Coroutine> childCoroutines = new List<Coroutine>();

            for (int j = 0; j < layerAct.activations.Count; j++)
            {
                float activation = layerAct.activations[j];

                if (activation >= activationThreshold)
                {
                    GameObject targetNeuron = atlasGenerator.GetNeuron(layerAct.name, j);

                    if (targetNeuron != null)
                    {
                        Coroutine c = StartCoroutine(SpawnAndAnimateChildOrb(
                            mainOrb.transform.position,
                            targetNeuron.transform.position,
                            activation,
                            layerAct.name,
                            j
                        ));
                        childCoroutines.Add(c);
                    }
                }
            }

            // Wait for all child orbs to reach their targets
            foreach (Coroutine c in childCoroutines)
            {
                yield return c;
            }

            yield return new WaitForSeconds(layerDelayTime * 0.5f);
        }

        // Fade out main orb
        yield return StartCoroutine(FadeOutOrb(mainOrb));

        isAnimating = false;
        Debug.Log("Animation complete! Press SPACE to replay or R to reset.");
    }

    IEnumerator MoveOrb(GameObject orb, Vector3 targetPos, float speed)
    {
        while (Vector3.Distance(orb.transform.position, targetPos) > 0.1f)
        {
            orb.transform.position = Vector3.MoveTowards(
                orb.transform.position,
                targetPos,
                speed * Time.deltaTime
            );
            yield return null;
        }
        orb.transform.position = targetPos;
    }

    IEnumerator SpawnAndAnimateChildOrb(Vector3 startPos, Vector3 targetPos, float activation, string layerName, int neuronId)
    {
        // Create child orb
        GameObject childOrb = Instantiate(childOrbPrefab, startPos, Quaternion.identity);

        // Size based on activation strength
        float orbSize = Mathf.Lerp(minOrbSize, maxOrbSize, activation);
        ConfigureOrb(childOrb, orbSize);

        // Animate to target
        yield return StartCoroutine(MoveOrb(childOrb, targetPos, childOrbSpeed));

        // Replace neuron with glowing version
        atlasGenerator.ReplaceWithGlowingNeuron(layerName, neuronId, activation);

        // Destroy child orb
        Destroy(childOrb);
    }

    void ConfigureOrb(GameObject orb, float size)
    {
        orb.transform.localScale = Vector3.one * size;
    }

    IEnumerator FadeOutOrb(GameObject orb)
    {
        float duration = 1f;
        float elapsed = 0f;

        Renderer renderer = orb.GetComponent<Renderer>();
        Light light = orb.GetComponent<Light>();

        if (renderer != null)
        {
            Color startColor = renderer.material.color;

            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float alpha = 1f - (elapsed / duration);

                Color newColor = startColor;
                newColor.a = alpha;
                renderer.material.color = newColor;

                if (light != null)
                {
                    light.intensity *= alpha;
                }

                yield return null;
            }
        }

        Destroy(orb);
    }

    void ResetVisualization()
    {
        isAnimating = false;

        if (mainOrb != null)
        {
            Destroy(mainOrb);
        }

        // Clear all spawned neurons and reload atlas
        if (atlasGenerator != null)
        {
            atlasGenerator.LoadAndGenerateAtlas();
        }

        // Reset camera to initial position if it exists
        if (mainCamera != null && followMainOrb)
        {
            mainCamera.transform.position = new Vector3(0, 30, -50);
            mainCamera.transform.rotation = Quaternion.Euler(20, 0, 0);
        }

        // Show the drawing canvas again when reset
        if (drawingInterface != null)
        {
            drawingInterface.ShowCanvas();
        }

        Debug.Log("Visualization reset. Press SPACE to start.");
    }
}