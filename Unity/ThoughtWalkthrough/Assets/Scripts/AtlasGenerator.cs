using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

[System.Serializable]
public class Neuron
{
    public int id;
    public float x;
    public float y;
    public float z;
}

[System.Serializable]
public class Layer
{
    public string name;
    public int neuron_count;
    public float z_offset;
    public List<Neuron> neurons;
}

[System.Serializable]
public class AtlasStructure
{
    public List<Layer> layers;
}

public class AtlasGenerator : MonoBehaviour
{
    [Header("References")]
    public GameObject neuronPrefab;
    public GameObject glowingNeuronPrefab;
    public Transform atlasParent;

    [Header("Files")]
    public string atlasJsonPath = "atlas_structure.json";

    [Header("Visualization Settings")]
    public float neuronScale = 0.5f;
    public bool spawnNeuronsImmediately = false;

    private AtlasStructure atlasData;
    private Dictionary<string, Dictionary<int, GameObject>> layerNeurons = new Dictionary<string, Dictionary<int, GameObject>>();
    private Dictionary<string, List<NeuronData>> layerNeuronData = new Dictionary<string, List<NeuronData>>();

    [System.Serializable]
    private class NeuronData
    {
        public int id;
        public Vector3 position;
    }

    void Start()
    {
        LoadAndGenerateAtlas();
    }

    public void LoadAndGenerateAtlas()
    {
        // Load JSON
        string fullPath = Path.Combine(Application.dataPath, atlasJsonPath);

        if (!File.Exists(fullPath))
        {
            Debug.LogError($"Atlas file not found: {fullPath}");
            return;
        }

        string jsonContent = File.ReadAllText(fullPath);
        atlasData = JsonUtility.FromJson<AtlasStructure>(jsonContent);

        if (atlasData == null || atlasData.layers == null)
        {
            Debug.LogError("Failed to parse atlas structure!");
            return;
        }

        Debug.Log($"Loaded {atlasData.layers.Count} layers");

        // Store neuron data for later spawning
        PrepareNeuronData();

        // Generate neurons immediately if enabled
        if (spawnNeuronsImmediately)
        {
            GenerateAllNeurons();
        }
    }

    void PrepareNeuronData()
    {
        layerNeuronData.Clear();

        foreach (Layer layer in atlasData.layers)
        {
            List<NeuronData> neurons = new List<NeuronData>();

            foreach (Neuron neuron in layer.neurons)
            {
                NeuronData data = new NeuronData
                {
                    id = neuron.id,
                    position = new Vector3(neuron.x, neuron.y, neuron.z)
                };
                neurons.Add(data);
            }

            layerNeuronData[layer.name] = neurons;
        }
    }

    void GenerateAllNeurons()
    {
        foreach (Layer layer in atlasData.layers)
        {
            GenerateLayerNeurons(layer.name);
        }
    }

    void GenerateNeurons()
    {
        // Legacy method - now calls GenerateAllNeurons
        GenerateAllNeurons();
    }

    public void GenerateLayerNeurons(string layerName)
    {
        // Check if layer already exists
        if (layerNeurons.ContainsKey(layerName))
        {
            Debug.Log($"Layer {layerName} neurons already spawned");
            return;
        }

        if (!layerNeuronData.ContainsKey(layerName))
        {
            Debug.LogError($"No data found for layer: {layerName}");
            return;
        }

        if (neuronPrefab == null)
        {
            Debug.LogError("Neuron prefab not assigned!");
            return;
        }

        if (atlasParent == null)
        {
            atlasParent = transform;
        }

        Dictionary<int, GameObject> neurons = new Dictionary<int, GameObject>();
        List<NeuronData> neuronDataList = layerNeuronData[layerName];

        foreach (NeuronData data in neuronDataList)
        {
            GameObject neuronObj = Instantiate(neuronPrefab, data.position, Quaternion.identity, atlasParent);
            neuronObj.name = $"{layerName}_Neuron_{data.id}";

            // Start with scale 0 for pop-in animation
            neuronObj.transform.localScale = Vector3.zero;
            StartCoroutine(PopInNeuron(neuronObj, neuronScale));

            neurons[data.id] = neuronObj;
        }

        layerNeurons[layerName] = neurons;
        Debug.Log($"Generated {neurons.Count} neurons for {layerName}");
    }

    IEnumerator PopInNeuron(GameObject neuron, float targetScale)
    {
        float duration = 0.3f;
        float elapsed = 0f;
        Vector3 target = Vector3.one * targetScale;

        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / duration;
            // Ease out elastic
            float scale = Mathf.Sin(t * Mathf.PI * 0.5f);
            neuron.transform.localScale = target * scale;
            yield return null;
        }

        neuron.transform.localScale = target;
    }

    // Public accessor for ThoughtProcess script
    public GameObject GetNeuron(string layerName, int neuronId)
    {
        if (layerNeurons.ContainsKey(layerName) && layerNeurons[layerName].ContainsKey(neuronId))
        {
            return layerNeurons[layerName][neuronId];
        }
        return null;
    }

    // Replace a neuron with its glowing version
    public void ReplaceWithGlowingNeuron(string layerName, int neuronId, float intensity)
    {
        if (!layerNeurons.ContainsKey(layerName) || !layerNeurons[layerName].ContainsKey(neuronId))
        {
            Debug.LogWarning($"Neuron not found: {layerName}, ID {neuronId}");
            return;
        }

        if (glowingNeuronPrefab == null)
        {
            Debug.LogError("Glowing neuron prefab not assigned!");
            return;
        }

        GameObject oldNeuron = layerNeurons[layerName][neuronId];
        Vector3 position = oldNeuron.transform.position;
        float scale = oldNeuron.transform.localScale.x;

        // Destroy old neuron
        Destroy(oldNeuron);

        // Spawn glowing neuron
        GameObject glowingNeuron = Instantiate(glowingNeuronPrefab, position, Quaternion.identity, atlasParent);
        glowingNeuron.name = $"{layerName}_Neuron_{neuronId}_Active";
        glowingNeuron.transform.localScale = Vector3.one * scale;

        // Update dictionary reference
        layerNeurons[layerName][neuronId] = glowingNeuron;
    }

    // Public accessor for checking if layer is spawned
    public bool IsLayerSpawned(string layerName)
    {
        return layerNeurons.ContainsKey(layerName);
    }

    public AtlasStructure GetAtlasData()
    {
        return atlasData;
    }
}