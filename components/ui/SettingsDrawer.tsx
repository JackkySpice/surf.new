'use client';

import { useEffect, useState } from 'react';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Settings } from 'lucide-react';
import {
  useSettings,
  AgentSettings,
  ModelSettings,
} from '@/app/contexts/SettingsContext';
import { cn } from '@/lib/utils';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Info } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useChatContext } from '@/app/contexts/ChatContext';
import { useSteelContext } from '@/app/contexts/SteelContext';

interface AgentConfig {
  name: string;
  description: string;
  supported_models: Array<{
    provider: string;
    models: string[];
  }>;
  model_settings: ModelConfig;
  agent_settings: {
    [key: string]: SettingConfig;
  };
}

interface AvailableAgents {
  [key: string]: AgentConfig;
}

interface ModelConfig {
  max_tokens: SettingConfig;
  temperature: SettingConfig;
  top_p: SettingConfig;
  top_k: SettingConfig;
  frequency_penalty: SettingConfig;
  presence_penalty: SettingConfig;
  [key: string]: SettingConfig; // Allow dynamic string keys
}

interface SettingConfig {
  type: 'integer' | 'float' | 'text' | 'textarea';
  default: number | string;
  min?: number;
  max?: number;
  step?: number;
  maxLength?: number;
  description?: string;
}

export function SettingsButton() {
  const { currentSettings } = useSettings();
  const [showSettings, setShowSettings] = useState(false);
  const router = useRouter();
  return (
    <Sheet open={showSettings} onOpenChange={setShowSettings}>
      <SheetTrigger asChild>
        <Button
          className={[
            'inline-flex items-center justify-center overflow-hidden gap-1.5',
            'h-8 pl-0 pr-2.5 rounded-full border text-sm font-normal leading-[14px]',
            'font-geist',
            'bg-[--gray-2] border-[--gray-3] text-[--gray-12]',
            'hover:bg-[--gray-3]',
            'disabled:text-[--gray-8]',
            '',
          ].join(' ')}
        >
          <div
            className={[
              'w-8 h-8 flex items-center justify-center gap-2.5 border-r text-sm',
              'bg-transparent',
              'border-[--gray-3]',
              'font-geist',
              'group-hover:bg-transparent',
            ].join(' ')}
          >
            <Settings className="w-5 h-5" />
          </div>
          {currentSettings?.selectedAgent}
        </Button>
      </SheetTrigger>
      <SettingsContent closeSettings={() => setShowSettings(false)} />
    </Sheet>
  );
}

function SettingInput({
  settingKey,
  config,
  value,
  onChange,
}: {
  settingKey: string;
  config: SettingConfig;
  value: any;
  onChange: (value: any) => void;
}) {
  const label = settingKey
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  // Use config.default if value is undefined
  const currentValue = value ?? config.default;

  // Sanitize number inputs
  const sanitizeNumber = (value: number) => {
    if (config.max !== undefined) value = Math.min(value, config.max);
    if (config.min !== undefined) value = Math.max(value, config.min);
    return value;
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label className="text-sm font-medium">{label}</label>
        {config.description && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Info className="h-4 w-4  rounded-full" />
              </TooltipTrigger>
              <TooltipContent className="bg-[--gray-1] text-[--gray-11] border border-[--gray-3]">
                {config.description}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>

      {config.type === 'float' && (
        <div className="flex flex-col gap-2">
          <Slider
            value={[currentValue]}
            min={config.min ?? 0}
            max={config.max ?? 1}
            step={config.step ?? 0.1}
            onValueChange={([newValue]) => onChange(sanitizeNumber(newValue))}
            className="[&_[role=slider]]:bg-[--gray-12] [&_[role=slider]]:border-[--gray-12] [&_[role=slider]]:shadow-sm
                       [&_.relative]:bg-[--gray-6] 
                       [&_[data-disabled]]:opacity-50
                       [&_[role=slider]]:focus:ring-2 [&_[role=slider]]:focus:ring-[--gray-8]
                       [&_[role=slider]]:focus-visible:outline-none
                       [&_.absolute]:bg-[--gray-12]"
          />
          <div className="text-sm text-[--gray-11] text-right">
            {Number(currentValue).toFixed(2)}
          </div>
        </div>
      )}

      {config.type === 'integer' && (
        <Input
          type="number"
          value={currentValue}
          min={config.min}
          max={config.max}
          step={1}
          onChange={(e) => {
            const newValue = parseInt(e.target.value);
            if (!isNaN(newValue)) {
              onChange(sanitizeNumber(newValue));
            }
          }}
          onBlur={(e) => {
            const newValue = parseInt(e.target.value);
            if (!isNaN(newValue)) {
              onChange(sanitizeNumber(newValue));
            } else {
              onChange(config.default);
            }
          }}
          className="settings-input"
        />
      )}

      {config.type === 'text' && (
        <Input
          type="text"
          value={currentValue}
          maxLength={config.maxLength}
          onChange={(e) => onChange(e.target.value)}
          className="settings-input"
        />
      )}

      {config.type === 'textarea' && (
        <Textarea
          value={currentValue}
          maxLength={config.maxLength}
          onChange={(e) => onChange(e.target.value)}
          className="settings-input min-h-[100px]"
        />
      )}
    </div>
  );
}

function SettingsContent({ closeSettings }: { closeSettings: () => void }) {
  const [agents, setAgents] = useState<AvailableAgents | null>(null);
  const { currentSettings, updateSettings } = useSettings();
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const { clearInitialState } = useChatContext();
  const { resetSession } = useSteelContext();

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<
    Array<{ tag: string; base_name: string }>
  >([]);
  const [isLoadingOllamaModels, setIsLoadingOllamaModels] = useState(false);
  const [ollamaError, setOllamaError] = useState<string | null>(null);

  const [selectedAgent, setSelectedAgent] = useState(
    currentSettings?.selectedAgent
  );
  const [selectedProvider, setSelectedProvider] = useState(
    currentSettings?.selectedProvider
  );

  const [selectedModel, setSelectedModel] = useState(
    currentSettings?.selectedModel
  );

  const [modelSettings, setModelSettings] = useState<ModelSettings | undefined>(
    currentSettings?.modelSettings
  );

  // Add state for agent settings
  const [agentSettings, setAgentSettings] = useState<AgentSettings | undefined>(
    currentSettings?.agentSettings
  );

  // Fetch Ollama models when provider is set to Ollama
  useEffect(() => {
    if (selectedProvider === 'ollama') {
      const fetchOllamaModels = async () => {
        try {
          setIsLoadingOllamaModels(true);
          setOllamaError(null);
          const response = await fetch('/api/ollama/models');

          if (!response.ok) {
            throw new Error(
              `Failed to fetch Ollama models: ${
                response.status
              } ${await response.text()}`
            );
          }

          const data = await response.json();
          setOllamaModels(data.models || []);

          // If we have models and the current selected model is not in the list,
          // select the first available model
          if (data.models && data.models.length > 0) {
            // Check if the current model is in the list of tags
            const modelTags = data.models.map(
              (model: { tag: string }) => model.tag
            );
            if (!modelTags.includes(selectedModel)) {
              setSelectedModel(data.models[0].tag);
            }
          }
        } catch (error) {
          console.error('Error fetching Ollama models:', error);
          setOllamaError(
            error instanceof Error
              ? error.message
              : 'Failed to fetch Ollama models'
          );
          // Fallback to default models from the agent config
          const defaultModels =
            (selectedAgent &&
              agents?.[selectedAgent]?.supported_models.find(
                (m: any) => m.provider === 'ollama'
              )?.models) ||
            [];
          setOllamaModels(
            defaultModels.map((model: any) => ({
              tag: model,
              base_name: model,
            }))
          );
        } finally {
          setIsLoadingOllamaModels(false);
        }
      };

      fetchOllamaModels();
    }
  }, [selectedProvider, selectedAgent, agents, selectedModel]);

  useEffect(() => {
    async function fetchAgents() {
      try {
        const response = await fetch('/api/agents');
        if (!response.ok) {
          throw new Error(
            `Failed to fetch agents: ${
              response.status
            } ${await response.text()}`
          );
        }
        const data: AvailableAgents = await response.json();
        setAgents(data);
        if (!currentSettings) {
          const firstAgentKey = Object.keys(data)[0];
          const defaultAgent = data[firstAgentKey].name;
          const defaultProvider =
            data[firstAgentKey].supported_models[0].provider;
          const defaultModel =
            data[firstAgentKey].supported_models[0].models[0];
          const defaultModelSettings = Object.entries(
            data[firstAgentKey].model_settings
          ).reduce((acc, [key, value]) => {
            acc[key] = value.default;
            return acc;
          }, {} as ModelSettings);
          const defaultAgentSettings = Object.entries(
            data[firstAgentKey].agent_settings
          ).reduce((acc, [key, value]) => {
            acc[key] = value.default;
            return acc;
          }, {} as AgentSettings);

          updateSettings({
            selectedAgent: firstAgentKey,
            selectedProvider: defaultProvider,
            selectedModel: defaultModel,
            modelSettings: defaultModelSettings,
            agentSettings: defaultAgentSettings,
            providerApiKeys: {}, // Start with empty object for new settings
          });
          setSelectedAgent(firstAgentKey);
          setSelectedProvider(defaultProvider);
          setSelectedModel(defaultModel);
          setModelSettings(defaultModelSettings);
          setAgentSettings(defaultAgentSettings);
        }
      } catch (error) {
        console.error('Error fetching agents:', error);
      } finally {
        setLoading(false);
      }
    }
    setLoading(true);
    fetchAgents();
  }, []);

  useEffect(() => {
    if (agents && selectedAgent) {
      setSelectedProvider(agents[selectedAgent].supported_models[0].provider);
      setSelectedModel(agents[selectedAgent].supported_models[0].models[0]);

      // Initialize model settings with all default values from the config
      const defaultModelSettings = Object.entries(
        agents[selectedAgent].model_settings
      ).reduce((acc, [key, config]) => {
        acc[key] = config.default;
        return acc;
      }, {} as ModelSettings);

      // Initialize agent settings with all default values from the config
      const defaultAgentSettings = Object.entries(
        agents[selectedAgent].agent_settings
      ).reduce((acc, [key, config]) => {
        acc[key] = config.default;
        return acc;
      }, {} as AgentSettings);

      setModelSettings(defaultModelSettings);
      setAgentSettings(defaultAgentSettings);
    }
  }, [selectedAgent, agents]);

  // Effect to update selected model when provider changes
  useEffect(() => {
    if (agents && selectedAgent && selectedProvider) {
      // Find the supported models for the selected provider
      const providerModels = agents[selectedAgent].supported_models.find(
        (m) => m.provider === selectedProvider
      );

      // If we found models for this provider, set the first one as default
      if (providerModels && providerModels.models.length > 0) {
        setSelectedModel(providerModels.models[0]);
      }
    }
  }, [selectedProvider, selectedAgent, agents]);

  // When setting agent/model settings, store only the values, not the config objects
  const handleSettingChange = (
    settingType: 'model' | 'agent',
    key: string,
    value: any
  ) => {
    if (settingType === 'model') {
      setModelSettings(
        (prev) =>
          ({
            ...(prev || {}),
            [key]: value,
          } as ModelSettings)
      );
    } else {
      setAgentSettings(
        (prev) =>
          ({
            ...(prev || {}),
            [key]: value,
          } as AgentSettings)
      );
    }
  };

  if (!selectedAgent) {
    return null;
  }

  function handleSave() {
    if (
      !selectedAgent ||
      !selectedProvider ||
      !selectedModel ||
      !modelSettings ||
      !agentSettings
    ) {
      return;
    }

    // Update settings first
    updateSettings({
      selectedAgent,
      selectedProvider,
      selectedModel,
      modelSettings,
      agentSettings,
      providerApiKeys: currentSettings?.providerApiKeys || {},
    });

    // Clear all state
    clearInitialState();
    resetSession();

    // Close settings drawer
    closeSettings();

    // Navigate to home page
    router.push('/');
  }

  if (loading || !agents) {
    return (
      <div className="flex items-center justify-center w-3 h-3">
        <div className="w-3 h-3 border-2 border-[--gray-8] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <SheetContent
      className={cn(
        'flex w-1/3 h-full flex-col shrink-0 min-w-[380px] max-w-full',
        'rounded-[20px] border border-[--gray-3] bg-[--gray-1]',
        'shadow-[0_16px_32px_-12px_rgba(14,18,27,0.10)] p-6 text-[--gray-12]'
      )}
    >
      <SheetHeader>
        <SheetTitle>Agent Settings</SheetTitle>
        <SheetDescription className="text-sm text-[--gray-11]">
          Configure which agent you want to use and fine-tune their behavior.
        </SheetDescription>
      </SheetHeader>

      {/* Add a scrollable container with styled scrollbar */}
      <div
        className="flex-1 overflow-y-auto my-6 pr-4 
        [&::-webkit-scrollbar]:w-2.5
        [&::-webkit-scrollbar-track]:bg-[--gray-3]
        [&::-webkit-scrollbar-track]:rounded-full
        [&::-webkit-scrollbar-thumb]:bg-[--gray-7]
        [&::-webkit-scrollbar-thumb]:rounded-full
        [&::-webkit-scrollbar-thumb]:border-4
        [&::-webkit-scrollbar-thumb]:hover:bg-[--gray-8]
        [&::-webkit-scrollbar-thumb]:transition-colors
        scrollbar-gutter-stable
        scrollbar-thin
        space-y-4"
      >
        {/* Agent Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Agent</label>
          <Select
            value={selectedAgent}
            onValueChange={(value) => setSelectedAgent(value)}
            disabled={Object.keys(agents).length === 0}
          >
            <SelectTrigger className="settings-input">
              <SelectValue placeholder="Select an agent" />
            </SelectTrigger>
            <SelectContent className="settings-input bg-[--gray-1] text-[--gray-11] border border-[--gray-3]">
              {Object.entries(agents).map(([key, agent]) => (
                <SelectItem key={key} value={key}>
                  {agent.name} - {agent.description}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-4">
          {/* Model Selection */}
          {agents[selectedAgent]?.supported_models && (
            <div className="space-y-2">
              <div className="space-y-2">
                <label className="text-sm font-medium">Model Provider</label>
                <Select
                  value={
                    selectedProvider ??
                    agents[selectedAgent].supported_models[0].provider
                  }
                  onValueChange={(value) => {
                    setSelectedProvider(value);
                  }}
                >
                  <SelectTrigger className="settings-input">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="settings-input">
                    {agents[selectedAgent].supported_models.map(
                      (supportedModel) => (
                        <SelectItem
                          key={supportedModel.provider}
                          value={supportedModel.provider}
                        >
                          {supportedModel.provider}
                        </SelectItem>
                      )
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Model</label>
                <Select
                  value={
                    selectedModel ??
                    agents[selectedAgent].supported_models[0].models[0]
                  }
                  onValueChange={(value) => setSelectedModel(value)}
                >
                  <SelectTrigger className="settings-input">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="settings-input">
                    {selectedProvider === 'ollama' ? (
                      isLoadingOllamaModels ? (
                        <SelectItem value="loading" disabled>
                          Loading Ollama models...
                        </SelectItem>
                      ) : ollamaError ? (
                        <>
                          <SelectItem value="error" disabled>
                            Error loading models
                          </SelectItem>
                          {agents[selectedAgent].supported_models
                            .find((m) => m.provider === selectedProvider)
                            ?.models.map((model) => (
                              <SelectItem key={model} value={model}>
                                {model} (fallback)
                              </SelectItem>
                            ))}
                        </>
                      ) : ollamaModels.length > 0 ? (
                        ollamaModels.map((model) => (
                          <SelectItem key={model.tag} value={model.tag}>
                            {model.tag}
                          </SelectItem>
                        ))
                      ) : (
                        <SelectItem value="no-models" disabled>
                          No models found. Install models with 'ollama pull'
                        </SelectItem>
                      )
                    ) : (
                      agents[selectedAgent].supported_models
                        .find((m) => m.provider === selectedProvider)
                        ?.models.map((model) => (
                          <SelectItem key={model} value={model}>
                            {model}
                          </SelectItem>
                        ))
                    )}
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}

          {/* API Key Management */}
          {selectedProvider && selectedProvider !== 'ollama' && (
            <div className="space-y-2 pt-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">
                  {selectedProvider} API Key
                </label>
                {currentSettings?.providerApiKeys?.[selectedProvider] && (
                  <button
                    onClick={() => {
                      // Clear the API key for this provider
                      const newKeys = { ...currentSettings.providerApiKeys };
                      delete newKeys[selectedProvider];
                      updateSettings({
                        ...currentSettings,
                        providerApiKeys: newKeys,
                      });
                    }}
                    className="text-sm text-[--gray-11] hover:text-[--gray-12]"
                  >
                    Clear Key
                  </button>
                )}
              </div>
              <div className="relative">
                <Input
                  type="password"
                  placeholder={
                    currentSettings?.providerApiKeys?.[selectedProvider]
                      ? '••••••••••••••••'
                      : `Enter ${selectedProvider} API Key`
                  }
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="settings-input pr-20"
                />
                {apiKey && (
                  <button
                    onClick={() => {
                      // Save the API key
                      const currentKeys =
                        currentSettings?.providerApiKeys || {};
                      updateSettings({
                        ...currentSettings!,
                        providerApiKeys: {
                          ...currentKeys,
                          [selectedProvider]: apiKey,
                        },
                      });
                      setApiKey(''); // Clear input after saving
                    }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 
                             px-3 py-1 text-sm bg-[--gray-3] hover:bg-[--gray-4] 
                             text-[--gray-12] rounded-full transition-colors"
                  >
                    Save
                  </button>
                )}
              </div>
              <p className="text-sm text-[--gray-11]">
                Your API key will be stored locally and never shared
              </p>
            </div>
          )}

          {/* Ollama Instructions */}
          {selectedProvider === 'ollama' && (
            <div className="space-y-2 pt-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Ollama Setup</label>
              </div>
              <div className="p-3 bg-[--gray-3] rounded-md">
                <p className="text-sm text-[--gray-12] mb-2">
                  Ollama runs locally on your machine and doesn't require an API
                  key.
                </p>
                <p className="text-sm text-[--gray-11] mb-2">
                  1. Install Ollama from{' '}
                  <a
                    href="https://ollama.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[--blue-11] hover:underline"
                  >
                    ollama.com
                  </a>
                </p>
                <p className="text-sm text-[--gray-11] mb-2">
                  2. Run Ollama locally with the model of your choice:
                  <code className="block mt-1 p-1 bg-[--gray-4] rounded text-xs">
                    ollama run {selectedModel || 'MODEL_NAME'}
                  </code>
                </p>
                <p className="text-sm text-[--gray-11]">
                  3. Surf.new will connect to your local Ollama instance
                  automatically
                </p>
                {ollamaError && (
                  <div className="mt-2 p-2 bg-[--red-3] border border-[--red-6] rounded-md text-[--red-11] text-xs">
                    Error: {ollamaError}. Make sure Ollama is running.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Advanced Settings */}
          <div className="space-y-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAdvanced((prev) => !prev)}
              className={[
                'w-full inline-flex items-center justify-between',
                'h-8 px-2.5 rounded-full border text-sm font-normal leading-[14px]',
                'font-geist',
                'bg-transparent border-transparent text-[--gray-12]',
                'hover:bg-[--gray-3]',
                'disabled:text-[--gray-8]',
              ].join(' ')}
            >
              Advanced Settings
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className={cn(
                  'transform transition-transform duration-200',
                  showAdvanced ? 'rotate-180' : ''
                )}
              >
                <path
                  d="M2.5 4.5L6 8L9.5 4.5"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </Button>
            {showAdvanced && (
              <div className="mt-2 space-y-4 animate-in slide-in-from-top-2 duration-200">
                {/* Model Settings */}
                <div className="space-y-4">
                  <h3 className="text-sm font-medium">Model Settings</h3>
                  {Object.entries(agents[selectedAgent].model_settings).map(
                    ([key, config]) => (
                      <SettingInput
                        key={key}
                        settingKey={key}
                        config={config}
                        value={modelSettings?.[key] ?? config.default}
                        onChange={(value) =>
                          handleSettingChange('model', key, value)
                        }
                      />
                    )
                  )}
                </div>

                {/* Agent Settings */}
                {Object.entries(agents[selectedAgent].agent_settings).map(
                  ([key, config]) => (
                    <SettingInput
                      key={key}
                      settingKey={key}
                      config={config}
                      value={agentSettings?.[key] ?? config.default}
                      onChange={(value) =>
                        handleSettingChange('agent', key, value)
                      }
                    />
                  )
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Keep the button at the bottom */}
      <div className="mt-auto">
        <div
          className="h-8 px-2 py-1 bg-[--gray-12] rounded-full justify-center items-center inline-flex overflow-hidden cursor-pointer"
          onClick={handleSave}
        >
          <div className="px-1 justify-start items-start flex">
            <div className="text-neutral-900 text-sm font-medium font-geist leading-normal">
              Apply Changes & Restart Chat
            </div>
          </div>
        </div>
      </div>
    </SheetContent>
  );
}
