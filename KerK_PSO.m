%Aum Sri Sai Ram
%This Program Combines Kernel-Kmeans & PSO 

dimensions=2;
K_no=2;  %No of Clusters
KMAX_ITER=10;   % no if K-Means iterations
PMAX_ITER=20;  % no if PSO iterations
No_particles=10; %no of swarm_particles
%meas=meas(:,3:3+dimensions-1); %data set chosen
meas=dlmread('jain.txt');
meas=meas(:,1:2);
%meas=meas+3;
plot(meas(:,1),meas(:,2),'*');
axis equal;
hold on
cent=rand(K_no,dimensions); %randomly generated centroids
data_range=max(meas)-min(meas);
cent=cent.*repmat(data_range,K_no,1) + repmat(min(meas),K_no,1)
datasize=size(meas);
plot(cent(1,1),cent(1,2),'ro');
plot(cent(2,1),cent(2,2),'ro');
centroid=0;



%calculating the distances to initialise the assignment if clusterpoints to
%nearest centriod (just initialisation)
for centroid= 1:K_no
    for data=1 : datasize(1)
        dist(data,centroid)=norm(cent(centroid,:)-meas(data,:));
    end
end

%assigning initial centriods
[value, index] = min(dist,[],2);
d = index;


for iter = 1 : KMAX_ITER
    % calculating the kernalised dist
    K_dist=zeros(datasize(1),K_no);
    no_of_data=0;
    temp=0;
    if(any(d==1) && any(d == 2))      %checking for good initialisation
        for centroid = 1 : K_no
            temp=0;
            no_of_data=0;
            for j = 1 : datasize(1)
                for l = 1 : datasize(1)
                    if(d(j) == centroid && d(l) == centroid)
                        no_of_data=no_of_data+1;
                        temp = temp + Kernal(meas(j,:),meas(l,:));
                    end
                end
            end
            term3 = temp/no_of_data;
            
            for i = 1 : datasize(1)
                term1=Kernal(meas(i,:),meas(i,:));
                no_of_data=0;    %initialie to zero for each cluster j
                temp=0;
                for j = 1 : datasize(1)
                    if(d(j)==centroid)
                        no_of_data=no_of_data+1;
                        temp=temp+Kernal(meas(i,:),meas(j,:));
                    end
                end
                term2=2*(temp/no_of_data);
                
                K_dist(i,centroid)=term1-term2+term3;    %calculated the distance if centriod from datapoint in feature space
            end
        end
    else
        fprintf('Bad initialisation');
    end
    
    %updating the cluster centriods
    [value index]=min(K_dist,[],1);
    for centroid = 1 : K_no
        cent(centroid,:)=meas(index(centroid),:);
    end
    
    %updating the assigment of points to the clusters
    [value, K_index] = min(K_dist,[],2);
    d = K_index;
end  %end of iterations

%PSO from here

swarm_particle=rand(K_no,dimensions,No_particles);
swarm_velocity=rand(K_no,dimensions,No_particles)*0.1;
data_range=max(meas)-min(meas);
swarm_particle=swarm_particle.*repmat(data_range,K_no,1,No_particles) + repmat(min(meas),K_no,1,No_particles)
datasize=size(meas);

%initialising swarm particle to K-Means centroid
swarm_particle(:,:,1)=cent;



%initialise p_best to Inf
p_best=Inf(No_particles,1);
pbest_location=zeros(K_no,dimensions,No_particles);
global_best_particle=zeros(K_no,dimensions);
w=0.72;
c1=1.49;
c2=1.49;

for particle = 1 : No_particles
    for centroid= 1:K_no
        for data=1 : datasize(1)
            dist(data,centroid,particle)=norm(swarm_particle(centroid,:,particle)-meas(data,:));
        end
    end
end

%assigning initial centriods
for particle = 1 : No_particles
    [value, index] = min(dist(:,:,particle),[],2);
    d(:,particle) = index;
end

for iter = 1 : PMAX_ITER
    % calculating the kernalised dist
    K_dist=zeros(datasize(1),K_no,No_particles);
    no_of_data=0;
    temp=0;
    for particle = 1 : No_particles
    if(any(d(:,particle)==1) && any(d(:,particle)== 2))    %checking for good initialisation
        for centroid = 1 : K_no
            temp=0;
            no_of_data=0;
            for j = 1 : datasize(1)
                for l = 1 : datasize(1)
                    if(d(j,particle) == centroid && d(l,particle) == centroid)
                        no_of_data=no_of_data+1;
                        temp = temp + Kernal(meas(j,:),meas(l,:));
                    end
                end
            end
            term3 = temp/no_of_data;
            
            for i = 1 : datasize(1)
                term1=Kernal(meas(i,:),meas(i,:));
                no_of_data=0;    %initialie to zero for each cluster j
                temp=0;
                for j = 1 : datasize(1)
                    if(d(j,particle)==centroid)
                        no_of_data=no_of_data+1;
                        temp=temp+Kernal(meas(i,:),meas(j,:));
                    end
                end
                term2=2*(temp/no_of_data);
                
                K_dist(i,centroid,particle)=term1-term2+term3;    %calculated the distance if centriod from datapoint in feature space
            end
        end
    else
        fprintf('Bad initialisation');
    end
    
    %updating the assigment of points to the clusters
    [value, index] = min(K_dist(:,:,particle),[],2);
    d(:,particle) = index;
    
    %Evaluation of the objective function
    for particle = 1:No_particles
        local_fitness=sum(sum(K_dist(:,:,particle)));
        if(p_best(particle,1)>local_fitness)
           p_best(particle,1)=local_fitness;
           pbest_location(:,:,particle)=swarm_particle(:,:,particle);
        end
    end
    
    %looking for the global best particle and saving the position as
    %global_best_particle
    [value g_index] = min(p_best,[],1);
    global_best_particle=swarm_particle(:,:,g_index);
    
    %going for the velocity update
    r1=rand;
    r2=rand;
    for i = 1:No_particles
        temp=0;
        inertial_term = w * swarm_velocity(:,:,i);
        cognitive_term = c1 * r1 * (pbest_location(:,:,i)-swarm_particle(:,:,i));
        social_term = c2 * r2 * (global_best_particle-swarm_particle(:,:,i));
        temp = inertial_term + cognitive_term + social_term;
        swarm_particle(:,:,i) = swarm_particle(:,:,i) + temp ; % UPDATE PARTICLE POSE
        swarm_velocity(:,:,i) = temp; % UPDATE PARTICLE VEL
    end
    
    end %end of the condition telling that no cluster is empty
    
    
end  %end of iterations


for centroid = 1:K_no
   if any(d(:)==centroid)
     if (centroid==1)
       plot(meas(d(:,g_index)==centroid,1),meas(d(:,g_index)==centroid,2),'cO','Markersize',4);
     else if(centroid==2)
         plot(meas(d(:,g_index)==centroid,1),meas(d(:,g_index)==centroid,2),'mO','Markersize',4);
     end
   end
   end
end
%Accuracy
idx=d(:,g_index);
meas_correct=dlmread('jain_correct.txt');
count_TP_TN=0;
for data=1:datasize(1)
    if (idx(data)==meas_correct(data))
        count_TP_TN=count_TP_TN+1;
    end
end
Accuracy=count_TP_TN/datasize(1);