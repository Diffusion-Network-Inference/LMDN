function [node_infected_states, node_infected_times] = generateDataFunction(network, node_num, infectRate, infectProb)

% 随机选择部分感染点
node_infected_times = ones(node_num,1) * (-1);
node_infected_states = zeros(node_num,1);

infected_node_num = ceil(node_num * infectRate);
infectedIndex = zeros(infected_node_num,1);
while (infected_node_num > 0)
    
    selectId = node_num * rand();
    selectId = ceil(selectId);
    if node_infected_states(selectId) == 0
        node_infected_states(selectId) = 1;
        infectedIndex(infected_node_num) = selectId;
        infected_node_num = infected_node_num - 1;
    end
    
end
fprintf('initial infected nodes have been selected\n');

% 感染传播
infectTime = 0;
bbreak = false;
last_node_infected = zeros(node_num,1);
ite = 0;
while (~bbreak)
    new_infect = node_infected_states - last_node_infected;
    new_infect_id = find(new_infect == 1);
    node_infected_times(new_infect_id) = infectTime;
    infectTime = infectTime + 1;
    if size(new_infect_id, 1) == 0
        bbreak = true;
    else
        ite = ite + 1;
        fprintf('infectIteration = %d, infectNum = %d\n', ite, size(new_infect_id, 1));
        last_node_infected = node_infected_states;
        for i = 1 : size(new_infect_id,1)
            infect_id = new_infect_id(i);
            infect_id_neighbor = network(network(:,1) == infect_id,2);
            for j = 1 : size(infect_id_neighbor,1)
                neighbor_j = infect_id_neighbor(j);
                if node_infected_states(neighbor_j) == 0
                    randP = rand();
                    if randP < infectProb
                        node_infected_states(neighbor_j) = 1;
                    end
                end
                
            end
            
        end
    end
end
fprintf('infect is done\n')
end

