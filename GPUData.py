class GPUData(BaseModel):
    gpu_utilization__pct: float = None 
    memory_utilization__pct: float = None  
    gpu_power_draw__W: float = None  
    gpu_temperature__C: float = None 
    gpu_fan_speed__pct: float = None  
    gpu_clock_speed__MHz: float = None  
    cpu_utilization__pct: float = None  
    memory_usage__pct: float = None  
    server_power_draw__W: float = None  
    server_temperature__C: float = None 
    disk_usage__pct: float = None 
    network_bandwidth__Mbps: float = None  