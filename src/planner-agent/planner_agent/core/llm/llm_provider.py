# Copyright (C) 2025 StructBinary
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os
from typing import Optional, Dict, Any
from langchain_core.runnables import Runnable
from planner_agent.utils.exceptions import UnsupportedProviderError, LLMConfigurationError


class LLMProvider:
    """Factory for creating LangChain-compatible LLM instances that always return Runnables."""
    
    # Supported providers registry
    _SUPPORTED_PROVIDERS = {
        "openai", 
        "anthropic", 
        "azure_openai"
    }
    
    @staticmethod
    def create_llm(
        provider: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs: Any
    ) -> Runnable:
        """
        Create a LangChain LLM instance that implements the Runnable interface.
        
        Args:
            provider: LLM provider name ('openai', 'anthropic', 'azure_openai')
            model: Model name (e.g., 'gpt-4', 'claude-3-sonnet-20240229')
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Configured LangChain LLM instance (guaranteed to be a Runnable)
            
        Raises:
            UnsupportedProviderError: If provider is not supported
            LLMConfigurationError: If configuration is invalid
        """
        provider = provider.lower().strip()
        
        if provider not in LLMProvider._SUPPORTED_PROVIDERS:
            supported = ", ".join(LLMProvider._SUPPORTED_PROVIDERS)
            raise UnsupportedProviderError(
                f"Unsupported provider: '{provider}'. "
                f"Supported providers: {supported}"
            )
        
        try:
            return LLMProvider._create_provider_instance(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
        except ImportError as e:
            raise LLMConfigurationError(
                f"Required package not installed for provider '{provider}': {e}"
            )
        except Exception as e:
            raise LLMConfigurationError(
                f"Failed to create LLM for provider '{provider}': {e}"
            )
    
    @staticmethod
    def _check_package(package_name: str, provider_name: str) -> None:
        """Check if required package is installed."""
        try:
            __import__(package_name)
        except ImportError:
            raise ImportError(
                f"{package_name} package is required for {provider_name} provider. "
                f"Install with: pip install {package_name}"
            )
    
    @staticmethod
    def _create_provider_instance(
        provider: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: int,
        **kwargs: Any
    ) -> Runnable:
        """
        Create the actual provider instance using the factory pattern.
        All LangChain chat models are Runnables by default.
        """
        if provider == "openai":
            return LLMProvider._create_openai_llm(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
        elif provider == "anthropic":
            return LLMProvider._create_anthropic_llm(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
        elif provider == "azure_openai":
            return LLMProvider._create_azure_openai_llm(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
        else:
            # This should never happen due to validation above
            raise UnsupportedProviderError(f"Provider '{provider}' not implemented")
    
    @staticmethod
    def _create_openai_llm(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: int,
        **kwargs: Any
    ) -> Runnable:
        """Create OpenAI LLM instance (returns a Runnable)."""
        LLMProvider._check_package("langchain_openai", "OpenAI")
        
        from langchain_openai import ChatOpenAI
        
        # Get API key from kwargs or environment
        api_key = kwargs.pop('api_key', None) or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise LLMConfigurationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Build configuration with proper parameter names
        config = {
            "model": model,
            "temperature": temperature,
            "openai_api_key": api_key,
        }
        
        # Add optional parameters with correct naming
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if kwargs.get('base_url'):
            config["openai_api_base"] = kwargs.pop('base_url')
        if kwargs.get('organization'):
            config["openai_organization"] = kwargs.pop('organization')
        
        # Add any remaining kwargs
        config.update(kwargs)
        
        # ChatOpenAI is a Runnable by default
        return ChatOpenAI(**config)
    
    @staticmethod
    def _create_anthropic_llm(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: int,
        **kwargs: Any
    ) -> Runnable:
        """Create Anthropic LLM instance (returns a Runnable)."""
        LLMProvider._check_package("langchain_anthropic", "Anthropic")
        
        from langchain_anthropic import ChatAnthropic  # type: ignore
        
        # Get API key from kwargs or environment
        api_key = kwargs.pop('api_key', None) or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise LLMConfigurationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Build configuration with proper parameter names
        config = {
            "model": model,
            "temperature": temperature,
            "anthropic_api_key": api_key,
        }
        
        # Add optional parameters with correct naming
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if kwargs.get('base_url'):
            config["base_url"] = kwargs.pop('base_url')
        
        # Add any remaining kwargs
        config.update(kwargs)
        
        # ChatAnthropic is a Runnable by default
        return ChatAnthropic(**config)
    
    @staticmethod
    def _create_azure_openai_llm(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: int,
        **kwargs: Any
    ) -> Runnable:
        """Create Azure OpenAI LLM instance (returns a Runnable)."""
        LLMProvider._check_package("langchain_openai", "Azure OpenAI")
        
        from langchain_openai import AzureChatOpenAI
        
        # Get configuration from kwargs or environment
        api_key = kwargs.pop('api_key', None) or os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = kwargs.pop('azure_endpoint', None) or os.getenv('AZURE_OPENAI_ENDPOINT')
        
        if not api_key:
            raise LLMConfigurationError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        if not endpoint:
            raise LLMConfigurationError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT environment variable "
                "or pass azure_endpoint parameter."
            )
        
        # Build configuration with proper parameter names
        config = {
            "azure_deployment": model,  # In Azure, this is the deployment name
            "temperature": temperature,
            "openai_api_key": api_key,
            "azure_endpoint": endpoint,
            "openai_api_version": kwargs.pop('api_version', "2024-02-15-preview"),
        }
        
        # Add optional parameters with correct naming
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        
        # Add any remaining kwargs
        config.update(kwargs)
        
        # AzureChatOpenAI is a Runnable by default
        return AzureChatOpenAI(**config)
    
    @staticmethod
    def get_supported_providers() -> Dict[str, Dict[str, str]]:
        """Get information about supported providers and their requirements."""
        return {
            "openai": {
                "description": "OpenAI GPT models",
                "required_env": "OPENAI_API_KEY",
                "package": "langchain-openai",
                "example_models": "gpt-4, gpt-4-turbo, gpt-3.5-turbo"
            },
            "anthropic": {
                "description": "Anthropic Claude models", 
                "required_env": "ANTHROPIC_API_KEY",
                "package": "langchain-anthropic",
                "example_models": "claude-3-opus-20240229, claude-3-sonnet-20240229"
            },
            "azure_openai": {
                "description": "Azure OpenAI Service",
                "required_env": "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT",
                "package": "langchain-openai", 
                "example_models": "gpt-4, gpt-35-turbo (deployment names)"
            }
        }
    
    @staticmethod
    def validate_environment(provider: str) -> Dict[str, bool]:
        """
        Validate that required environment variables are set for a provider.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            Dictionary with validation results
        """
        provider = provider.lower().strip()
        validation = {"valid": True, "missing": []}
        
        if provider == "openai":
            if not os.getenv('OPENAI_API_KEY'):
                validation["valid"] = False
                validation["missing"].append("OPENAI_API_KEY")
        
        elif provider == "anthropic":
            if not os.getenv('ANTHROPIC_API_KEY'):
                validation["valid"] = False
                validation["missing"].append("ANTHROPIC_API_KEY")
        
        elif provider == "azure_openai":
            if not os.getenv('AZURE_OPENAI_API_KEY'):
                validation["valid"] = False
                validation["missing"].append("AZURE_OPENAI_API_KEY")
            if not os.getenv('AZURE_OPENAI_ENDPOINT'):
                validation["valid"] = False
                validation["missing"].append("AZURE_OPENAI_ENDPOINT")
        
        else:
            validation["valid"] = False
            validation["missing"] = [f"Unsupported provider: {provider}"]
        
        return validation


# Convenience function for quick LLM creation
def create_llm_from_env(
    provider: str = "openai", 
    model: str = "gpt-4",
    **kwargs: Any
) -> Runnable:
    """
    Convenience function to create an LLM using environment variables.
    
    Args:
        provider: LLM provider name (default: 'openai')
        model: Model name (default: 'gpt-4')
        **kwargs: Additional parameters for LLM configuration
        
    Returns:
        Configured LangChain LLM instance (guaranteed to be a Runnable)
    """
    return LLMProvider.create_llm(provider=provider, model=model, **kwargs) 