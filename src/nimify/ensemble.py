"""Model ensemble orchestration for advanced multi-model inference."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass 
class EnsembleConfig:
    """Configuration for model ensemble."""
    name: str
    strategy: str
    models: dict[str, Any]  
    routing_rules: dict[str, Any]
    timeout_seconds: float = 30.0
    max_parallel: int = 5


class EnsembleRequest(BaseModel):
    """Request for ensemble inference."""
    input: list[list[float]] = Field(..., description="Input data")
    strategy_override: str | None = Field(None, description="Override ensemble strategy")
    model_preferences: list[str] | None = Field(None, description="Preferred model order")


class EnsembleResponse(BaseModel):
    """Response from ensemble inference.""" 
    predictions: list[list[float]] = Field(..., description="Aggregated predictions")
    model_results: dict[str, Any] = Field(..., description="Individual model results")
    inference_time_ms: float = Field(..., description="Total inference time")
    strategy_used: str = Field(..., description="Strategy that was applied")


class ModelEnsemble:
    """Orchestrates multiple NIM services for advanced inference patterns."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self._performance_cache = {}
        
    async def predict(self, request: EnsembleRequest) -> EnsembleResponse:
        """Run ensemble prediction with intelligent routing."""
        start_time = time.time()
        strategy = request.strategy_override or self.config.strategy
        
        try:
            if strategy == "sequential":
                results = await self._sequential_inference(request)
            elif strategy == "parallel":
                results = await self._parallel_inference(request)
            elif strategy == "conditional":
                results = await self._conditional_inference(request)
            elif strategy == "load_balance":
                results = await self._load_balanced_inference(request)
            else:
                raise ValueError(f"Unknown ensemble strategy: {strategy}")
            
            # Aggregate results
            aggregated_predictions = self._aggregate_predictions(results, strategy)
            
            total_time_ms = (time.time() - start_time) * 1000
            
            return EnsembleResponse(
                predictions=aggregated_predictions,
                model_results=results,
                inference_time_ms=total_time_ms,
                strategy_used=strategy
            )
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Ensemble error: {str(e)}")
    
    async def _sequential_inference(self, request: EnsembleRequest) -> dict[str, Any]:
        """Execute models in sequence, passing output of one as input to next."""
        results = {}
        current_input = request.input
        
        model_order = self.config.routing_rules.get("order", list(self.config.models.keys()))
        
        for model_name in model_order:
            if model_name not in self.config.models:
                logger.warning(f"Model {model_name} not found in ensemble")
                continue
                
            model = self.config.models[model_name]
            start_time = time.time()
            
            try:
                # Use previous output as input for next model
                model_predictions = await model.predict(current_input)
                inference_time = (time.time() - start_time) * 1000
                
                results[model_name] = {
                    "predictions": model_predictions,
                    "inference_time_ms": inference_time,
                    "status": "success"
                }
                
                # Update input for next model
                current_input = model_predictions
                
            except Exception as e:
                results[model_name] = {
                    "error": str(e),
                    "inference_time_ms": (time.time() - start_time) * 1000,
                    "status": "error"
                }
                # Break chain on error
                break
        
        return results
    
    async def _parallel_inference(self, request: EnsembleRequest) -> dict[str, Any]:
        """Execute all models in parallel and aggregate results."""
        tasks = []
        model_names = []
        
        # Create tasks for parallel execution
        for model_name, model in self.config.models.items():
            task = self._run_single_model(model_name, model, request.input)
            tasks.append(task)
            model_names.append(model_name)
        
        # Execute with timeout
        try:
            results_list = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error("Parallel inference timed out")
            return {"error": "Timeout occurred during parallel inference"}
        
        # Combine results
        results = {}
        for model_name, result in zip(model_names, results_list):
            if isinstance(result, Exception):
                results[model_name] = {
                    "error": str(result),
                    "status": "error"
                }
            else:
                results[model_name] = result
        
        return results
    
    async def _conditional_inference(self, request: EnsembleRequest) -> dict[str, Any]:
        """Route to specific model based on input characteristics."""
        # Analyze input to determine routing
        input_characteristics = self._analyze_input(request.input)
        
        selected_model = self._select_model_by_condition(
            input_characteristics,
            self.config.routing_rules.get("rules", [])
        )
        
        if selected_model not in self.config.models:
            raise ValueError(f"Selected model {selected_model} not available")
        
        # Run inference on selected model
        model = self.config.models[selected_model]
        result = await self._run_single_model(selected_model, model, request.input)
        
        return {selected_model: result}
    
    async def _load_balanced_inference(self, request: EnsembleRequest) -> dict[str, Any]:
        """Select model based on current load and performance metrics."""
        # Get current load metrics
        model_loads = await self._get_model_loads()
        
        # Select best model based on load and weights
        weights = self.config.routing_rules.get("weights", {})
        selected_model = self._select_least_loaded_model(model_loads, weights)
        
        if selected_model not in self.config.models:
            # Fallback to first available model
            selected_model = next(iter(self.config.models.keys()))
        
        model = self.config.models[selected_model]
        result = await self._run_single_model(selected_model, model, request.input)
        
        return {selected_model: result}
    
    async def _run_single_model(self, model_name: str, model: Any, input_data: list[list[float]]) -> dict[str, Any]:
        """Execute inference on a single model."""
        start_time = time.time()
        
        try:
            predictions = await model.predict(input_data)
            inference_time = (time.time() - start_time) * 1000
            
            # Update performance cache
            self._performance_cache[model_name] = {
                "last_inference_time": inference_time,
                "last_update": time.time()
            }
            
            return {
                "predictions": predictions,
                "inference_time_ms": inference_time,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "inference_time_ms": (time.time() - start_time) * 1000,
                "status": "error"
            }
    
    def _analyze_input(self, input_data: list[list[float]]) -> dict[str, Any]:
        """Analyze input characteristics for conditional routing."""
        input_array = np.array(input_data)
        
        return {
            "input_size": len(input_data),
            "feature_count": len(input_data[0]) if input_data else 0,
            "mean_value": float(np.mean(input_array)) if input_array.size > 0 else 0.0,
            "std_value": float(np.std(input_array)) if input_array.size > 0 else 0.0,
            "has_negative": bool(np.any(input_array < 0)) if input_array.size > 0 else False
        }
    
    def _select_model_by_condition(self, characteristics: dict[str, Any], rules: list[dict]) -> str:
        """Select model based on conditional rules."""
        for rule in rules:
            condition = rule.get("condition", "")
            
            if condition == "default":
                return rule["target"]
            
            # Simple condition evaluation
            if "input_size" in condition:
                size_threshold = int(condition.split("<")[1].strip())
                if characteristics["input_size"] < size_threshold:
                    return rule["target"]
        
        # Fallback to first model if no conditions match
        return next(iter(self.config.models.keys()))
    
    async def _get_model_loads(self) -> dict[str, float]:
        """Get current load metrics for each model."""
        loads = {}
        current_time = time.time()
        
        for model_name in self.config.models.keys():
            # Use cached performance metrics
            perf_data = self._performance_cache.get(model_name, {})
            last_inference_time = perf_data.get("last_inference_time", 100.0)  # Default 100ms
            last_update = perf_data.get("last_update", current_time - 3600)  # 1 hour ago default
            
            # Calculate load factor (lower is better)
            age_factor = min((current_time - last_update) / 300, 1.0)  # Age out after 5 minutes
            load_factor = last_inference_time * (1 + age_factor)
            
            loads[model_name] = load_factor
        
        return loads
    
    def _select_least_loaded_model(self, model_loads: dict[str, float], weights: dict[str, float]) -> str:
        """Select the model with the lowest weighted load."""
        best_model = None
        best_score = float('inf')
        
        for model_name, load in model_loads.items():
            weight = weights.get(model_name, 1.0)
            # Lower score is better (combines load and weight)
            score = load / weight
            
            if score < best_score:
                best_score = score
                best_model = model_name
        
        return best_model or next(iter(self.config.models.keys()))
    
    def _aggregate_predictions(self, results: dict[str, Any], strategy: str) -> list[list[float]]:
        """Aggregate predictions from multiple models."""
        valid_results = []
        
        # Collect valid predictions
        for model_name, result in results.items():
            if result.get("status") == "success" and "predictions" in result:
                valid_results.append(result["predictions"])
        
        if not valid_results:
            raise ValueError("No valid predictions from any model")
        
        if strategy == "parallel":
            # Weighted average aggregation
            return self._weighted_average_predictions(valid_results)
        else:
            # For sequential, conditional, and load_balance, return last valid result
            return valid_results[-1]
    
    def _weighted_average_predictions(self, predictions_list: list[list[list[float]]]) -> list[list[float]]:
        """Compute weighted average of predictions."""
        if not predictions_list:
            return []
        
        # Convert to numpy for easier computation
        arrays = [np.array(pred) for pred in predictions_list]
        
        # Simple average (could be enhanced with learned weights)
        averaged = np.mean(arrays, axis=0)
        
        return averaged.tolist()
    
    def generate_ensemble_api(self) -> FastAPI:
        """Generate FastAPI app for ensemble service."""
        app = FastAPI(
            title=f"{self.config.name} Ensemble API",
            description="Multi-model ensemble inference service",
            version="1.0.0"
        )
        
        @app.post("/v1/ensemble/predict", response_model=EnsembleResponse)
        async def ensemble_predict(request: EnsembleRequest):
            """Run ensemble prediction."""
            return await self.predict(request)
        
        @app.get("/ensemble/status")
        async def ensemble_status():
            """Get ensemble status and model health."""
            model_status = {}
            
            for model_name in self.config.models.keys():
                perf_data = self._performance_cache.get(model_name, {})
                model_status[model_name] = {
                    "available": True,  # Simplified check
                    "last_inference_time_ms": perf_data.get("last_inference_time", 0),
                    "last_update": perf_data.get("last_update", 0)
                }
            
            return {
                "ensemble_name": self.config.name,
                "strategy": self.config.strategy,
                "models": model_status,
                "total_models": len(self.config.models)
            }
        
        return app