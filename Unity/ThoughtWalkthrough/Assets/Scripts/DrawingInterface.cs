using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.IO;

public class DrawingInterface : MonoBehaviour
{
    [Header("UI References")]
    public RawImage drawingCanvas;
    public Button clearButton;
    public Button submitButton;
    public Text instructionText;
    public GameObject canvasPanel;

    [Header("Drawing Settings")]
    public int canvasSize = 280;
    public float brushSize = 15f;
    public Color brushColor = Color.white;

    [Header("Python Configuration")]
    [Tooltip("The name of your python script inside the Assets folder")]
    public string pythonScriptName = "mnist_inference.py";

    [Tooltip("The name of your model file inside the Assets folder")]
    public string modelFileName = "mnist_model.pth";

    [Tooltip("Absolute path to your Python executable")]
    public string pythonExecutablePath = @"C:\Users\cgall\Documents\GitHub\ThoughtWalkthrough\.venv\Scripts\python.exe";

    private Texture2D drawingTexture;
    private bool isDrawing = false;
    private Vector2 lastMousePos;
    private ThoughtProcess thoughtProcess;

    void Start()
    {
        // Create drawing texture
        drawingTexture = new Texture2D(canvasSize, canvasSize);
        ClearCanvas();
        drawingCanvas.texture = drawingTexture;

        // Setup buttons
        clearButton.onClick.AddListener(ClearCanvas);
        submitButton.onClick.AddListener(OnSubmitDrawing);

        // Get ThoughtProcess reference
        thoughtProcess = FindObjectOfType<ThoughtProcess>();

        instructionText.text = "Draw a digit (0-9) and click Submit";
    }

    void Update()
    {
        // Check if mouse is over canvas
        RectTransform canvasRect = drawingCanvas.GetComponent<RectTransform>();
        Vector2 localMousePos;

        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(
            canvasRect, Input.mousePosition, null, out localMousePos))
        {
            Vector2 normalizedPos = new Vector2(
                (localMousePos.x + canvasRect.rect.width / 2) / canvasRect.rect.width,
                (localMousePos.y + canvasRect.rect.height / 2) / canvasRect.rect.height
            );

            Vector2 texturePos = new Vector2(
                normalizedPos.x * canvasSize,
                normalizedPos.y * canvasSize
            );

            if (Input.GetMouseButtonDown(0))
            {
                isDrawing = true;
                lastMousePos = texturePos;
                DrawAtPosition(texturePos);
            }
            else if (Input.GetMouseButton(0) && isDrawing)
            {
                DrawLine(lastMousePos, texturePos);
                lastMousePos = texturePos;
            }
            else if (Input.GetMouseButtonUp(0))
            {
                isDrawing = false;
            }
        }
        else
        {
            if (Input.GetMouseButtonUp(0))
            {
                isDrawing = false;
            }
        }
    }

    void DrawAtPosition(Vector2 pos)
    {
        int x = Mathf.RoundToInt(pos.x);
        int y = Mathf.RoundToInt(pos.y);
        int radius = Mathf.RoundToInt(brushSize / 2);

        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                if (i * i + j * j <= radius * radius)
                {
                    int px = x + i;
                    int py = y + j;

                    if (px >= 0 && px < canvasSize && py >= 0 && py < canvasSize)
                    {
                        drawingTexture.SetPixel(px, py, brushColor);
                    }
                }
            }
        }
        drawingTexture.Apply();
    }

    void DrawLine(Vector2 start, Vector2 end)
    {
        float distance = Vector2.Distance(start, end);
        int steps = Mathf.CeilToInt(distance);

        for (int i = 0; i <= steps; i++)
        {
            float t = i / (float)steps;
            Vector2 point = Vector2.Lerp(start, end, t);
            DrawAtPosition(point);
        }
    }

    void ClearCanvas()
    {
        Color clearColor = Color.black;
        for (int x = 0; x < canvasSize; x++)
        {
            for (int y = 0; y < canvasSize; y++)
            {
                drawingTexture.SetPixel(x, y, clearColor);
            }
        }
        drawingTexture.Apply();
    }

    void OnSubmitDrawing()
    {
        instructionText.text = "Processing...";
        submitButton.interactable = false;
        StartCoroutine(ProcessDrawing());
    }

    IEnumerator ProcessDrawing()
    {
        // Save drawing as PNG
        byte[] pngData = drawingTexture.EncodeToPNG();
        string imagePath = Path.Combine(Application.dataPath, "user_drawing.png");
        File.WriteAllBytes(imagePath, pngData);

        Debug.Log($"Saved drawing to: {imagePath}");

        // Run Python script
        yield return StartCoroutine(RunPythonInference(imagePath));

        // Wait a moment for files to be written
        yield return new WaitForSeconds(0.5f);

        // Read the result file directly here to show the log
        string jsonPath = Path.Combine(Application.dataPath, "inference_trace.json");
        if (File.Exists(jsonPath))
        {
            try
            {
                string jsonContent = File.ReadAllText(jsonPath);
                InferenceTrace traceData = JsonUtility.FromJson<InferenceTrace>(jsonContent);
                Debug.Log($"<color=cyan>The System thought you drew a: {traceData.predicted_label}</color>");
            }
            catch
            {
                // Silently fail if structure isn't available or parse fails
            }
        }

        // Reload inference data in ThoughtProcess
        if (thoughtProcess != null)
        {
            thoughtProcess.ReloadInferenceData();
            instructionText.text = "Ready! Press SPACE to visualize";
            submitButton.interactable = true;
        }
        else
        {
            instructionText.text = "Error: ThoughtProcess not found!";
            submitButton.interactable = true;
        }
    }

    IEnumerator RunPythonInference(string imagePath)
    {
        // 1. Determine Script Path
        string scriptFullPath;

        // check if the path in the inspector is already an absolute path (like C:/Users/...)
        if (Path.IsPathRooted(pythonScriptName))
        {
            scriptFullPath = pythonScriptName;
        }
        else
        {
            // If not, assume it's in the Assets folder
            scriptFullPath = Path.Combine(Application.dataPath, pythonScriptName);
        }

        string modelFullPath = Path.Combine(Application.dataPath, modelFileName);

        // 2. Validate Paths
        if (!File.Exists(pythonExecutablePath))
        {
            Debug.LogError($"Python executable not found at: {pythonExecutablePath}");
            instructionText.text = "Error: Python exe missing";
            submitButton.interactable = true;
            yield break;
        }

        if (!File.Exists(scriptFullPath))
        {
            Debug.LogError($"Python script not found at: {scriptFullPath}");
            Debug.LogError($"Make sure the path in Inspector matches: {scriptFullPath}");
            instructionText.text = "Error: Script missing";
            submitButton.interactable = true;
            yield break;
        }

        // 3. Prepare Arguments
        // Note: Your manual CLI command only used 2 args (script + image). 
        // We will send model path only if the script expects it, but sending it as a 3rd arg 
        // usually won't break Python unless the script strict argument checking.

        string arguments = $"\"{scriptFullPath}\" \"{imagePath}\" \"{modelFullPath}\"";

        Debug.Log($"Running Python: {pythonExecutablePath} {arguments}");

        // 4. Execute Process
        System.Diagnostics.Process process = new System.Diagnostics.Process();
        process.StartInfo.FileName = pythonExecutablePath;
        process.StartInfo.Arguments = arguments;
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.CreateNoWindow = true;

        process.Start();

        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();

        process.WaitForExit();

        Debug.Log($"Python output: {output}");
        if (!string.IsNullOrEmpty(error))
        {
            // Python writes warnings to stderr sometimes, so only error if exit code != 0
            if (process.ExitCode != 0)
            {
                Debug.LogError($"Python Error (Code {process.ExitCode}): {error}");
            }
            else
            {
                Debug.LogWarning($"Python Log: {error}");
            }
        }

        yield return null;
    }

    public void HideCanvas()
    {
        if (canvasPanel != null) canvasPanel.SetActive(false);
        else
        {
            drawingCanvas.gameObject.SetActive(false);
            clearButton.gameObject.SetActive(false);
            submitButton.gameObject.SetActive(false);
            instructionText.gameObject.SetActive(false);
        }
    }

    public void ShowCanvas()
    {
        if (canvasPanel != null) canvasPanel.SetActive(true);
        else
        {
            drawingCanvas.gameObject.SetActive(true);
            clearButton.gameObject.SetActive(true);
            submitButton.gameObject.SetActive(true);
            instructionText.gameObject.SetActive(true);
        }
    }
}