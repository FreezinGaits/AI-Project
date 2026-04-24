"""Add a model export cell to the notebook before the conclusion."""
import json

with open('Next_Location_Prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

export_md = {
    "cell_type": "markdown", "metadata": {},
    "source": ["## 10.5 Export Model for Production (FastAPI)\n",
               "Save the best model and encoders so our FastAPI backend can load them."]
}

export_code = {
    "cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
    "source": [
        "import joblib, os\n",
        "\n",
        "os.makedirs('model_artifacts', exist_ok=True)\n",
        "\n",
        "# Pick best ML model\n",
        "best_name = max(\n",
        "    {k: v for k, v in results.items() if k != 'Markov Chain'},\n",
        "    key=lambda k: results[k]['accuracy']\n",
        ")\n",
        "best_model = results[best_name]['model']\n",
        "print(f'Best model: {best_name} (acc={results[best_name][\"accuracy\"]:.4f})')\n",
        "\n",
        "# Save model + encoders + markov matrix\n",
        "joblib.dump(best_model, 'model_artifacts/best_model.pkl')\n",
        "joblib.dump(le_dict, 'model_artifacts/label_encoders.pkl')\n",
        "joblib.dump(le_target, 'model_artifacts/target_encoder.pkl')\n",
        "joblib.dump(feature_cols, 'model_artifacts/feature_cols.pkl')\n",
        "joblib.dump(markov_matrix, 'model_artifacts/markov_matrix.pkl')\n",
        "\n",
        "# Save all model results for comparison\n",
        "model_accuracies = {k: v['accuracy'] for k, v in results.items()}\n",
        "joblib.dump(model_accuracies, 'model_artifacts/model_accuracies.pkl')\n",
        "\n",
        "print('Saved to model_artifacts/')\n",
        "print(os.listdir('model_artifacts'))"
    ]
}

# Insert before the last cell (conclusion)
nb['cells'].insert(-1, export_md)
nb['cells'].insert(-1, export_code)

with open('Next_Location_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
